import pytest
import json
import hashlib
from pathlib import Path
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


@pytest.mark.asyncio
async def test_closed_candidate_is_not_linked_after_model_call(db_session_factory, db_clean):
    from app.flywheel.canonicalize import canonicalize_batch
    from app.db.models import CanonicalQuestion

    cid = await repository.create_conversation("alice")
    await repository.insert_low_confidence(cid, "猫粮可以退吗", "self_check", "无证据")
    async with db_session_factory.begin() as session:
        candidate = CanonicalQuestion(canonical_question="猫粮退货问题")
        session.add(candidate)
        await session.flush()
        candidate_id = candidate.id

    class ClosingModel(FakeStructuredModel):
        async def ainvoke(self, prompt):
            async with db_session_factory.begin() as session:
                row = await session.get(CanonicalQuestion, candidate_id)
                row.status = "rejected"
            return await super().ainvoke(prompt)

    model = ClosingModel([{"canonical_question": "猫粮退货问题", "draft_answer": "", "matched_question_id": candidate_id,
                           "reason": "候选"}])
    result = await canonicalize_batch(1, model)
    assert result.processed == 0
    async with db_session_factory() as session:
        assert await session.scalar(text("SELECT COUNT(*) FROM canonical_occurrences")) == 0


def test_masking_removes_identifiers():
    from app.flywheel.privacy import mask_sensitive

    result = mask_sensitive("订单 A123456，电话 13812345678，邮箱 a@example.com")
    assert "A123456" not in result and "13812345678" not in result and "a@example.com" not in result


@pytest.mark.asyncio
async def test_old_exact_canonical_key_is_reused_beyond_candidate_window(db_session_factory, db_clean):
    from app.flywheel.canonicalize import canonicalize_batch
    from app.flywheel.privacy import normalize_question
    from app.db.models import CanonicalQuestion

    question = "开封猫粮能退吗"
    key = hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()
    async with db_session_factory.begin() as session:
        original = CanonicalQuestion(canonical_question=question, canonical_key=key)
        session.add(original)
        await session.flush()
        original_id = original.id
        session.add_all([CanonicalQuestion(canonical_question=f"填充问题 {n}") for n in range(31)])
    cid = await repository.create_conversation("alice")
    await repository.insert_low_confidence(cid, "猫粮开封了可退吗", "self_check", "无证据")
    model = FakeStructuredModel([{"canonical_question": question, "draft_answer": "", "matched_question_id": None,
                                  "reason": "同一问题"}])
    result = await canonicalize_batch(1, model)
    assert result.merged == 1
    async with db_session_factory() as session:
        assert await session.scalar(text("SELECT canonical_id FROM canonical_occurrences")) == original_id


def test_labeled_sample_set_covers_required_distinctions():
    from app.flywheel.privacy import mask_sensitive, normalize_question

    labels = json.loads((Path(__file__).parent / "canonicalization-labels.json").read_text(encoding="utf-8"))
    assert {row["kind"] for row in labels} == {
        "same_question", "different_policy", "pii", "invalid_candidate",
    }
    same = [row for row in labels if row["kind"] == "same_question"]
    assert len({row["canonical"] for row in same}) == 1
    different = next(row for row in labels if row["kind"] == "different_policy")
    assert normalize_question(different["canonical"]) != normalize_question(same[0]["canonical"])
    pii = next(row for row in labels if row["kind"] == "pii")
    assert "13812345678" not in mask_sensitive(pii["question"])
