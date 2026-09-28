import pytest
from sqlalchemy import text

from app.db.models import CanonicalQuestion


@pytest.mark.asyncio
async def test_vector_failure_preserves_approved_decision(db_session_factory, db_clean, monkeypatch):
    from app.flywheel import review

    async with db_session_factory.begin() as session:
        row = CanonicalQuestion(canonical_question='手机13812345678咨询退款', draft_answer='草稿',
                                status='approved', category='售后', approved_answer='由售后核实商品状态。')
        session.add(row)
        await session.flush()
        identity = row.id
    published = await review.publish_approved(identity, 'publish-1')

    async def failure():
        raise ConnectionError('embedding unavailable')

    monkeypatch.setattr(review, 'vectorize_pending_knowledge', failure)
    with pytest.raises(ConnectionError):
        await review.retry_review_vectorization(identity, 'retry-1')
    detail = await review.get_review_detail(identity)
    assert detail['status'] == 'approved_pending_vector'
    assert detail['vector_status'] == 'pending'
    assert '13812345678' not in str(detail)

    async def success():
        async with db_session_factory.begin() as session:
            await session.execute(text("UPDATE knowledge_chunks SET vectorize_status='done' WHERE id=:id"),
                                  {'id': published.knowledge_chunk_id})
        return 1

    monkeypatch.setattr(review, 'vectorize_pending_knowledge', success)
    await review.retry_review_vectorization(identity, 'retry-1')
    assert (await review.get_review_detail(identity))['publication_status'] == 'vectorized'
    async with db_session_factory() as session:
        assert await session.scalar(text('SELECT COUNT(*) FROM knowledge_chunks')) == 1


@pytest.mark.asyncio
async def test_unapproved_review_cannot_retry_vectors(db_session_factory, db_clean, monkeypatch):
    from app.flywheel import review

    async with db_session_factory.begin() as session:
        row = CanonicalQuestion(canonical_question='问题', status='pending_review')
        session.add(row)
        await session.flush()
        identity = row.id
    async def forbidden():
        raise AssertionError('unapproved item called vectorization')
    monkeypatch.setattr(review, 'vectorize_pending_knowledge', forbidden)
    with pytest.raises(ValueError):
        await review.retry_review_vectorization(identity, 'retry-unapproved')
    assert await review.get_review_detail(999999) is None
