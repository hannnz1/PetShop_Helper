import pytest
from sqlalchemy import text

from app.db.models import CanonicalQuestion


async def _canonical(db_session_factory, *, approved=False):
    async with db_session_factory.begin() as session:
        row = CanonicalQuestion(canonical_question="猫粮可以退货吗？", draft_answer="模型草稿，不得发布",
                                status="approved" if approved else "pending_review",
                                category="售后" if approved else None,
                                approved_answer="请提供商品状态，由售后核实。" if approved else None)
        session.add(row)
        await session.flush()
        return row.id


@pytest.mark.asyncio
async def test_publish_requires_approval_and_reuses_knowledge(db_session_factory, db_clean):
    from app.flywheel.review import publish_approved

    unapproved = await _canonical(db_session_factory)
    with pytest.raises(ValueError):
        await publish_approved(unapproved, "pub-0")
    approved = await _canonical(db_session_factory, approved=True)
    first = await publish_approved(approved, "pub-1")
    second = await publish_approved(approved, "pub-1")
    assert first.knowledge_chunk_id == second.knowledge_chunk_id
    assert first.status == "approved_pending_vector"
    async with db_session_factory() as session:
        rows = (await session.execute(text("SELECT questions, answer, vectorize_status FROM knowledge_chunks"))).all()
        assert rows == [("猫粮可以退货吗？", "请提供商品状态，由售后核实。", "pending")]
        assert await session.scalar(text("SELECT COUNT(*) FROM flywheel_review_actions WHERE action='publish'")) == 1


@pytest.mark.asyncio
async def test_publish_recovers_after_knowledge_insert_before_link(db_session_factory, db_clean, monkeypatch):
    from app.flywheel import review

    canonical_id = await _canonical(db_session_factory, approved=True)
    original = review._link_published_chunk
    called = False

    async def fail_once(*args, **kwargs):
        nonlocal called
        if not called:
            called = True
            raise RuntimeError("simulated crash")
        return await original(*args, **kwargs)

    monkeypatch.setattr(review, "_link_published_chunk", fail_once)
    with pytest.raises(RuntimeError):
        await review.publish_approved(canonical_id, "pub-retry")
    recovered = await review.publish_approved(canonical_id, "pub-retry")
    assert recovered.knowledge_chunk_id > 0
    async with db_session_factory() as session:
        assert await session.scalar(text("SELECT COUNT(*) FROM knowledge_chunks")) == 1
