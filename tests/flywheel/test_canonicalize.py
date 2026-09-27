import pytest
from sqlalchemy import text

from app.db import repository


class FakeStructuredModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        return self

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return self.schema(**self.outputs.pop(0))


@pytest.mark.asyncio
async def test_canonicalize_merges_and_rerun_is_idempotent(db_session_factory, db_clean):
    from app.flywheel.canonicalize import canonicalize_batch

    conversation_id = await repository.create_conversation("alice")
    one = await repository.insert_low_confidence(conversation_id, "猫粮开封后还能退吗", "self_check", "无证据")
    two = await repository.insert_low_confidence(conversation_id, "开封的猫粮能退货吗", "retrieval_low_conf", "无证据")
    model = FakeStructuredModel([
        {"canonical_question": "开封猫粮可以退货吗？", "draft_answer": "待核实", "matched_question_id": None, "reason": "新问题"},
        {"canonical_question": "开封猫粮可以退货吗？", "draft_answer": "待核实", "matched_question_id": None, "reason": "同义问法"},
    ])
    result = await canonicalize_batch(10, model)
    assert (result.processed, result.new, result.merged) == (2, 1, 1)
    assert (await canonicalize_batch(10, model)).processed == 0
    async with db_session_factory() as session:
        rows = (await session.execute(text("SELECT canonical_id, raw_question_id FROM canonical_occurrences ORDER BY raw_question_id"))).all()
    assert [r.raw_question_id for r in rows] == [one, two]
    assert rows[0].canonical_id == rows[1].canonical_id
    assert len(model.prompts) == 2


@pytest.mark.asyncio
async def test_invalid_model_candidate_does_not_merge(db_session_factory, db_clean):
    from app.flywheel.canonicalize import canonicalize_batch

    cid = await repository.create_conversation("alice")
    await repository.insert_low_confidence(cid, "猫粮什么时候到", "self_check", "无证据")
    model = FakeStructuredModel([{"canonical_question": "猫粮何时到？", "draft_answer": "", "matched_question_id": 999999, "reason": "猜测"}])
    result = await canonicalize_batch(1, model)
    assert result.pending_upstream == 1
    async with db_session_factory() as session:
        assert await session.scalar(text("SELECT COUNT(*) FROM canonical_occurrences")) == 0


@pytest.mark.asyncio
async def test_external_guard_prevents_call(db_session_factory, db_clean, monkeypatch):
    from app.flywheel.canonicalize import canonicalize_batch

    cid = await repository.create_conversation("alice")
    await repository.insert_low_confidence(cid, "订单 A123456 什么时候到，电话 13812345678", "self_check", "无证据")
    monkeypatch.setenv("CHAT_BASE_URL", "https://external.example/v1")
    from app.config import get_settings
    get_settings.cache_clear()
    model = FakeStructuredModel([])
    result = await canonicalize_batch(5, model)
    assert result.pending_upstream == 1
    assert model.prompts == []
    get_settings.cache_clear()


def test_masking_removes_identifiers():
    from app.flywheel.privacy import mask_sensitive

    result = mask_sensitive("订单 A123456，电话 13812345678，邮箱 a@example.com")
    assert "A123456" not in result and "13812345678" not in result and "a@example.com" not in result
