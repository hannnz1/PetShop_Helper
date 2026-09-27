import pytest
from sqlalchemy import text


async def _question(db_session_factory, question="猫粮能退吗"):
    async with db_session_factory.begin() as session:
        row = __import__("app.db.models", fromlist=["CanonicalQuestion"]).CanonicalQuestion(canonical_question=question)
        session.add(row)
        await session.flush()
        return row.id


@pytest.mark.asyncio
async def test_review_requires_human_answer_and_is_request_idempotent(db_session_factory, db_clean):
    from app.flywheel.review import review_question

    question_id = await _question(db_session_factory)
    with pytest.raises(ValueError):
        await review_question(question_id, "approve", "req-1", category="售后")
    first = await review_question(question_id, "approve", "req-2", category="售后", approved_answer="请联系售后核实")
    same = await review_question(question_id, "approve", "req-2", category="售后", approved_answer="请联系售后核实")
    assert first.status == same.status == "approved"
    with pytest.raises(ValueError):
        await review_question(question_id, "approve", "req-2", category="售后", approved_answer="另一个答案")
    async with db_session_factory() as session:
        assert await session.scalar(text("SELECT COUNT(*) FROM flywheel_review_actions")) == 1


@pytest.mark.asyncio
async def test_review_transition_and_merge_target(db_session_factory, db_clean):
    from app.flywheel.review import review_question

    source = await _question(db_session_factory)
    target = await _question(db_session_factory, "宠物用品能退吗")
    with pytest.raises(ValueError):
        await review_question(source, "merge", "req-a", merge_target_id=999999)
    result = await review_question(source, "merge", "req-b", merge_target_id=target)
    assert result.status == "rejected" and result.merged_into_id == target
    with pytest.raises(ValueError):
        await review_question(source, "approve", "req-c", category="售后", approved_answer="可以")


@pytest.mark.asyncio
async def test_review_http_bearer_configured_and_missing(db_session_factory, db_clean, monkeypatch):
    from httpx import ASGITransport, AsyncClient
    from app.config import get_settings
    from app.main import app

    monkeypatch.delenv("KNOWLEDGE_REVIEW_TOKEN", raising=False)
    get_settings.cache_clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/api/review/questions")).status_code == 503
    monkeypatch.setenv("KNOWLEDGE_REVIEW_TOKEN", "review-test-token")
    get_settings.cache_clear()
    await _question(db_session_factory)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/api/review/questions")).status_code == 401
        assert (await client.get("/api/review/questions", headers={"Authorization": "Bearer wrong"})).status_code == 401
        ok = await client.get("/api/review/questions", headers={"Authorization": "Bearer review-test-token"})
        assert ok.status_code == 200
        assert ok.json()["items"][0]["canonical_question"] == "猫粮能退吗"
    get_settings.cache_clear()
