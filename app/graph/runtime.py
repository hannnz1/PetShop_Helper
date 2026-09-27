"""Own a persisted graph and serialize turns for each MySQL conversation."""

from collections.abc import AsyncIterator, Callable
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.config import Settings, get_settings
from app.core.context_layers import layer1_downgrade_boundary
from app.db import repository
from app.core.summarizer import schedule_summary
from app.graph.state import new_turn
from app.observability.tracing import make_turn_callbacks, turn_metadata
from app.observability.usage import UsageCollector
from app.db.observability import save_usage_events

logger = logging.getLogger(__name__)


class ConversationNotFound(Exception):
    """The conversation does not exist or belongs to another user."""


class ConversationBusy(Exception):
    """A turn for this conversation is already running in this worker."""


class GraphDivergence(Exception):
    """SQLite state and authoritative MySQL audit require operator recovery."""


class ConversationPending(Exception):
    """A user-confirmation interrupt must be resumed before a new chat turn."""


class ResumeNotPending(Exception):
    """This thread is not waiting for the requested user action."""


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

    def _turn_config(self, user_id: str, conversation_id: int) -> dict:
        """Allocate one ID for the entire Graph invocation or stream/resume."""
        turn_id = str(uuid4())
        logger.info("Graph turn started turn_id=%s conversation_id=%s", turn_id, conversation_id)
        settings = self.settings or get_settings()
        return {
            "configurable": {"thread_id": str(conversation_id)},
            "metadata": turn_metadata(settings, user_id, conversation_id, turn_id),
            "callbacks": make_turn_callbacks(settings, user_id, conversation_id, turn_id),
        }

    @staticmethod
    def _with_usage(config: dict, conversation_id: int) -> UsageCollector:
        collector = UsageCollector(config["metadata"]["turn_id"], conversation_id)
        config["callbacks"].append(collector)
        return collector

    @staticmethod
    async def _settle_usage(collector: UsageCollector, intent: str,
                            outcome: str) -> None:
        try:
            await save_usage_events(collector.finalize(intent, outcome=outcome))
        except Exception as exc:
            logger.warning("Usage persistence failed (%s)", type(exc).__name__)

    @staticmethod
    def _pending_kind(snapshot: Any, conversation_id: int) -> str | None:
        interrupts = getattr(snapshot, "interrupts", ())
        if (len(interrupts) != 1 or not isinstance(interrupts[0].value, dict)
                or snapshot.values.get("conversation_id") != conversation_id):
            return None
        kind = interrupts[0].value.get("type")
        expected_node = {"select_order": "fetch_order", "confirm_ticket": "agent_tools"}.get(kind)
        return kind if expected_node and tuple(snapshot.next) == (expected_node,) else None

    @staticmethod
    def _select_order_pending(snapshot: Any, conversation_id: int) -> bool:
        return GraphRuntime._pending_kind(snapshot, conversation_id) == "select_order"

    async def _check_audit(
        self, conversation_id: int, *, resume: bool = False,
        expected_user_id: str | None = None, expected_kind: str | None = None,
    ) -> None:
        if not self.enforce_audit:
            return
        config = {"configurable": {"thread_id": str(conversation_id)}}
        snapshot = await self.graph.aget_state(config)
        marker = (snapshot.values.get("trace") or {}).get("audit_message_id")
        latest = await repository.last_message_id(conversation_id)
        pending_kind = self._pending_kind(snapshot, conversation_id)
        pending = pending_kind is not None
        if (marker != latest or (snapshot.values and marker is None and not pending)
                or (expected_user_id is not None and snapshot.values
                    and snapshot.values.get("user_id") != expected_user_id)):
            logger.warning("Graph audit divergence conversation_id=%s checkpoint_marker=%s mysql_marker=%s pending=%s",
                           conversation_id, marker, latest, bool(snapshot.next))
            raise GraphDivergence(conversation_id)
        if snapshot.next:
            if pending:
                if resume:
                    if expected_kind != pending_kind:
                        raise ResumeNotPending(conversation_id)
                    return
                raise ConversationPending(conversation_id)
            raise GraphDivergence(conversation_id)
        if resume:
            raise ResumeNotPending(conversation_id)

    async def _prepare_context_snapshot(self, conversation_id: int, user_id: str):
        snapshot = await repository.get_context_snapshot(conversation_id, user_id)
        if snapshot is None:
            raise ConversationNotFound()
        # Before routing, reserve the most conservative route allocation. Only
        # audited pairs exist here: live tools and a pending user turn remain in
        # the checkpoint and cannot become a downgrade boundary.
        from app.graph.nodes import _budget
        settings = self.settings or get_settings()
        limit = min(_budget({"route": route}, snapshot, settings=settings).layer1
                    for route in ("business", "knowledge", "refund", "chitchat"))
        boundary = layer1_downgrade_boundary(snapshot, limit, settings)
        if boundary > snapshot.layer1_from_msg_id:
            await repository.advance_layer1(conversation_id, boundary)
            # Reload even after a lost race: the database boundary is authoritative.
            snapshot = await repository.get_context_snapshot(conversation_id, user_id)
            if snapshot is None:
                raise ConversationNotFound()
        return snapshot

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
            snapshot = await self._prepare_context_snapshot(resolved, user_id)
            if snapshot is None:
                raise ConversationNotFound()
            prior_marker = await repository.last_message_id(resolved)
            config = self._turn_config(user_id, resolved)
            collector = self._with_usage(config, resolved)
            try:
                result = await self.graph.ainvoke(
                    new_turn(user_id, resolved, message),
                    config,
                    context={"model": model, "classifier": self.classifier, "snapshot": snapshot,
                             "settings": self.settings, "turn_id": collector.turn_id},
                )
            except BaseException:
                await self._settle_usage(collector, "unknown", "failed")
                raise
            interrupted = "__interrupt__" in result
            await self._settle_usage(
                collector, result.get("intent") or "unknown",
                "interrupted" if interrupted else "completed",
            )
            final_marker = (result.get("trace") or {}).get("audit_message_id")
            if final_marker is not None and final_marker > (prior_marker or 0):
                layer2_limit = (result.get("trace") or {}).get("summary_layer2_budget")
                if layer2_limit is not None:
                    schedule_summary(resolved, layer2_token_limit=layer2_limit)
                else:
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
            snapshot = await self._prepare_context_snapshot(resolved, user_id)
            if snapshot is None:
                raise ConversationNotFound()
        except BaseException:
            self._active.remove(resolved)
            raise

        async def events() -> AsyncIterator[tuple[str, Any]]:
            collector = None
            intent = "unknown"
            outcome = "interrupted"
            saw_interrupt = False
            try:
                config = self._turn_config(user_id, resolved)
                collector = self._with_usage(config, resolved)
                async for event in self.graph.astream(
                    new_turn(user_id, resolved, message),
                    config,
                    context={"model": model, "classifier": self.classifier, "snapshot": snapshot,
                             "settings": self.settings, "turn_id": collector.turn_id},
                    stream_mode=["messages", "updates"],
                ):
                    if event[0] == "updates" and isinstance(event[1], dict):
                        saw_interrupt |= "__interrupt__" in event[1]
                        for update in event[1].values():
                            if isinstance(update, dict) and update.get("intent"):
                                intent = update["intent"]
                    yield event
                if not saw_interrupt:
                    outcome = "completed"
            except Exception:
                outcome = "failed"
                raise
            finally:
                try:
                    if collector is not None:
                        await self._settle_usage(collector, intent, outcome)
                finally:
                    self._active.remove(resolved)

        return events()

    async def prepare_resume_turn(
        self, user_id: str, conversation_id: int, resume_value: str | dict,
        *, model: BaseChatModel,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Authorize and verify a pending action frame before SSE headers."""
        if self.graph is None:
            raise RuntimeError("graph runtime is not open")
        resolved = await self._conversation_id(user_id, conversation_id)
        self._claim(resolved)
        try:
            expected_kind = "select_order" if isinstance(resume_value, str) else "confirm_ticket"
            if (expected_kind == "confirm_ticket"
                    and (not isinstance(resume_value, dict)
                         or type(resume_value.get("confirmed")) is not bool)):
                raise ResumeNotPending(resolved)
            await self._check_audit(resolved, resume=True, expected_user_id=user_id,
                                    expected_kind=expected_kind)
            snapshot = await self._prepare_context_snapshot(resolved, user_id)
            if snapshot is None:
                raise ConversationNotFound()
            pending_intent = "unknown"
            if hasattr(self.graph, "aget_state"):
                pending = await self.graph.aget_state(
                    {"configurable": {"thread_id": str(resolved)}})
                pending_intent = pending.values.get("intent") or "unknown"
        except BaseException:
            self._active.remove(resolved)
            raise

        async def events() -> AsyncIterator[tuple[str, Any]]:
            collector = None
            intent = pending_intent
            outcome = "interrupted"
            saw_interrupt = False
            try:
                config = self._turn_config(user_id, resolved)
                collector = self._with_usage(config, resolved)
                async for event in self.graph.astream(
                    Command(resume=resume_value),
                    config,
                    context={"model": model, "classifier": self.classifier, "snapshot": snapshot,
                             "settings": self.settings, "turn_id": collector.turn_id},
                    stream_mode=["messages", "updates"],
                ):
                    if event[0] == "updates" and isinstance(event[1], dict):
                        saw_interrupt |= "__interrupt__" in event[1]
                        for update in event[1].values():
                            if isinstance(update, dict) and update.get("intent"):
                                intent = update["intent"]
                    yield event
                if not saw_interrupt:
                    outcome = "completed"
            except Exception:
                outcome = "failed"
                raise
            finally:
                try:
                    if collector is not None:
                        await self._settle_usage(collector, intent, outcome)
                finally:
                    self._active.remove(resolved)

        return events()
