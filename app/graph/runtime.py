"""Own a persisted graph and serialize turns for each MySQL conversation."""

from collections.abc import AsyncIterator, Callable
import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.config import Settings
from app.db import repository
from app.core.summarizer import schedule_summary
from app.graph.state import new_turn

logger = logging.getLogger(__name__)


class ConversationNotFound(Exception):
    """The conversation does not exist or belongs to another user."""


class ConversationBusy(Exception):
    """A turn for this conversation is already running in this worker."""


class GraphDivergence(Exception):
    """SQLite state and authoritative MySQL audit require operator recovery."""


class ConversationPending(Exception):
    """A select-order interrupt must be resumed before a new chat turn."""


class ResumeNotPending(Exception):
    """This thread is not waiting for an order selection."""


class GraphRuntime:
    """One-worker runtime; the application owns its context-manager lifetime."""

    def __init__(self, checkpoint_path: Path,
                 graph_factory: Callable[[AsyncSqliteSaver], Any],
                 classifier: Any | None = None, *, enforce_audit: bool = True,
                 settings: Settings | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.graph_factory = graph_factory
        self.classifier = classifier
        self.enforce_audit = enforce_audit
        self.settings = settings
        self.graph: Any | None = None
        self._saver_context: Any | None = None
        self._active: set[int] = set()

    async def __aenter__(self) -> "GraphRuntime":
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        context = AsyncSqliteSaver.from_conn_string(str(self.checkpoint_path))
        saver = await context.__aenter__()
        try:
            self.graph = self.graph_factory(saver)
        except BaseException:
            await context.__aexit__(None, None, None)
            raise
        self._saver_context = context
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._saver_context is not None:
            await self._saver_context.__aexit__(exc_type, exc, traceback)
        self._saver_context = None
        self.graph = None

    async def _conversation_id(self, user_id: str, conversation_id: int | None) -> int:
        if conversation_id is None:
            return await repository.create_conversation(user_id)
        conversation = await repository.get_conversation(conversation_id)
        if conversation is None or conversation.user_id != user_id:
            raise ConversationNotFound()
        return conversation_id

    def _claim(self, conversation_id: int) -> None:
        if conversation_id in self._active:
            raise ConversationBusy()
        self._active.add(conversation_id)

    @staticmethod
    def _select_order_pending(snapshot: Any, conversation_id: int) -> bool:
        interrupts = getattr(snapshot, "interrupts", ())
        return (
            tuple(snapshot.next) == ("fetch_order",)
            and len(interrupts) == 1
            and isinstance(interrupts[0].value, dict)
            and interrupts[0].value.get("type") == "select_order"
            and snapshot.values.get("conversation_id") == conversation_id
        )

    async def _check_audit(
        self, conversation_id: int, *, resume: bool = False,
        expected_user_id: str | None = None,
    ) -> None:
        if not self.enforce_audit:
            return
        config = {"configurable": {"thread_id": str(conversation_id)}}
        snapshot = await self.graph.aget_state(config)
        marker = (snapshot.values.get("trace") or {}).get("audit_message_id")
        latest = await repository.last_message_id(conversation_id)
        pending = self._select_order_pending(snapshot, conversation_id)
        if (marker != latest or (snapshot.values and marker is None and not pending)
                or (expected_user_id is not None and snapshot.values
                    and snapshot.values.get("user_id") != expected_user_id)):
            logger.warning("Graph audit divergence conversation_id=%s checkpoint_marker=%s mysql_marker=%s pending=%s",
                           conversation_id, marker, latest, bool(snapshot.next))
            raise GraphDivergence(conversation_id)
        if snapshot.next:
            if pending:
                if resume:
                    return
                raise ConversationPending(conversation_id)
            raise GraphDivergence(conversation_id)
        if resume:
            raise ResumeNotPending(conversation_id)

    async def ainvoke_turn(
        self, user_id: str, message: str, conversation_id: int | None,
        *, model: BaseChatModel,
    ) -> dict:
        if self.graph is None:
            raise RuntimeError("graph runtime is not open")
        resolved = await self._conversation_id(user_id, conversation_id)
        self._claim(resolved)
        try:
            await self._check_audit(resolved, expected_user_id=user_id)
            snapshot = await repository.get_context_snapshot(resolved, user_id)
            if snapshot is None:
                raise ConversationNotFound()
            prior_marker = await repository.last_message_id(resolved)
            result = await self.graph.ainvoke(
                new_turn(user_id, resolved, message),
                {"configurable": {"thread_id": str(resolved)}},
                context={"model": model, "classifier": self.classifier, "snapshot": snapshot,
                         "settings": self.settings},
            )
            final_marker = (result.get("trace") or {}).get("audit_message_id")
            if final_marker is not None and final_marker > (prior_marker or 0):
                schedule_summary(resolved)
            return result
        finally:
            self._active.remove(resolved)

    async def astream_turn(
        self, user_id: str, message: str, conversation_id: int | None,
        *, model: BaseChatModel,
    ) -> AsyncIterator[tuple[str, Any]]:
        stream = await self.prepare_stream_turn(
            user_id, message, conversation_id, model=model,
        )
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    async def prepare_stream_turn(
        self, user_id: str, message: str, conversation_id: int | None,
        *, model: BaseChatModel,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Authorize and claim before HTTP headers; release when stream closes."""
        if self.graph is None:
            raise RuntimeError("graph runtime is not open")
        resolved = await self._conversation_id(user_id, conversation_id)
        self._claim(resolved)
        try:
            await self._check_audit(resolved, expected_user_id=user_id)
            snapshot = await repository.get_context_snapshot(resolved, user_id)
            if snapshot is None:
                raise ConversationNotFound()
        except BaseException:
            self._active.remove(resolved)
            raise

        async def events() -> AsyncIterator[tuple[str, Any]]:
            try:
                async for event in self.graph.astream(
                    new_turn(user_id, resolved, message),
                    {"configurable": {"thread_id": str(resolved)}},
                    context={"model": model, "classifier": self.classifier, "snapshot": snapshot,
                             "settings": self.settings},
                    stream_mode=["messages", "updates"],
                ):
                    yield event
            finally:
                self._active.remove(resolved)

        return events()

    async def prepare_resume_turn(
        self, user_id: str, conversation_id: int, order_id: str,
        *, model: BaseChatModel,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Authorize and verify a pending select-order frame before SSE headers."""
        if self.graph is None:
            raise RuntimeError("graph runtime is not open")
        resolved = await self._conversation_id(user_id, conversation_id)
        self._claim(resolved)
        try:
            await self._check_audit(resolved, resume=True, expected_user_id=user_id)
            snapshot = await repository.get_context_snapshot(resolved, user_id)
            if snapshot is None:
                raise ConversationNotFound()
        except BaseException:
            self._active.remove(resolved)
            raise

        async def events() -> AsyncIterator[tuple[str, Any]]:
            try:
                async for event in self.graph.astream(
                    Command(resume=order_id),
                    {"configurable": {"thread_id": str(resolved)}},
                    context={"model": model, "classifier": self.classifier, "snapshot": snapshot,
                             "settings": self.settings},
                    stream_mode=["messages", "updates"],
                ):
                    yield event
            finally:
                self._active.remove(resolved)

        return events()
