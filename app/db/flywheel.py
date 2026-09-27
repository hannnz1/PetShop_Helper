"""MySQL authority operations for the human-reviewed knowledge flywheel."""

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

import app.db.base as db
from app.db.models import Conversation, LowConfidenceQuestion, Message


async def record_unresolved_feedback(
    user_id: str, conversation_id: int, assistant_message_id: int,
) -> int:
    """Idempotently record feedback against an owned assistant answer."""
    source_ref = f"assistant:{assistant_message_id}"
    try:
        async with db.async_session.begin() as session:
            owned = await session.scalar(select(Conversation.id).where(
                Conversation.id == conversation_id, Conversation.user_id == user_id,
            ))
            answer = await session.scalar(select(Message).where(
                Message.id == assistant_message_id,
                Message.conversation_id == conversation_id,
                Message.role == "assistant",
            ))
            if owned is None or answer is None:
                raise LookupError("assistant message not found")
            question = await session.scalar(select(Message).where(
                Message.conversation_id == conversation_id,
                Message.role == "user",
                Message.id < assistant_message_id,
                Message.content.is_not(None),
            ).order_by(Message.id.desc()).limit(1))
            if question is None or not (question.content or "").strip():
                raise LookupError("preceding user question not found")
            prior = await session.scalar(select(LowConfidenceQuestion).where(
                LowConfidenceQuestion.source_ref == source_ref,
            ))
            if prior is not None:
                return prior.id
            row = LowConfidenceQuestion(
                conversation_id=conversation_id,
                raw_question=question.content,
                source="user_feedback",
                reason="用户反馈未解决",
                source_ref=source_ref,
            )
            session.add(row)
            await session.flush()
            return row.id
    except IntegrityError:
        # A concurrent repeat may win the unique source_ref race.
        async with db.async_session() as session:
            prior = await session.scalar(select(LowConfidenceQuestion).where(
                LowConfidenceQuestion.source_ref == source_ref,
                LowConfidenceQuestion.conversation_id == conversation_id,
            ))
            if prior is not None:
                return prior.id
        raise
