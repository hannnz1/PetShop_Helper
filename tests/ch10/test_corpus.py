import pytest


@pytest.mark.asyncio
async def test_external_model_guard_keeps_real_pool_local(monkeypatch):
    from app.config import get_settings
    from scripts.ch10.build_corpus import build_corpus

    monkeypatch.setenv("CHAT_BASE_URL", "https://external.example/v1")
    get_settings.cache_clear()
    calls = []

    async def prelabel(text):
        calls.append(text)
        return ("退换货",)

    result = await build_corpus([{"id": 7, "text": "订单202601180001234567想退款"}],
                                prelabel=prelabel, target_per_class=1)
    assert result.status == "pending_upstream" and not calls
    assert "202601180001234567" not in str(result)
    get_settings.cache_clear()


def test_multilabel_counts_and_seeded_audit_sample():
    from scripts.ch10.build_corpus import missing_by_class, audit_sample

    rows = [{"text": f"退货运费{i}", "labels": ["退换货", "运费"], "origin": "simulated"}
            for i in range(8)]
    missing = missing_by_class(rows, target=10)
    assert missing["退换货"] == 2 and missing["运费"] == 2
    assert audit_sample(rows, per_class=5, seed=42) == audit_sample(rows, per_class=5, seed=42)
    assert 5 <= len(audit_sample(rows, per_class=5, seed=42)) <= 8


def test_synthetic_only_export_uses_reviewed_labels(tmp_path):
    from scripts.ch10.build_corpus import run_synthetic

    report = run_synthetic(tmp_path)
    assert len(report.labeled) == 30
    assert (tmp_path / "corpus_labeled.jsonl").exists()
    assert (tmp_path / "audit.md").exists()


def test_failed_prelabel_still_enters_masked_human_review(tmp_path):
    import json
    from scripts.ch10.build_corpus import CorpusReport, write_corpus
    from scripts.ch10.review_corpus import apply_reviews

    report = CorpusReport("partial", failures=(18,), failed_rows=(
        {"id": 18, "text": "QQ号: [账号] 想退货", "origin": "pool",
         "prelabel_status": "failed"},))
    write_corpus(report, tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "corpus_labeled.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["review_id"] == "pool-18"
    assert "pool-18" in (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert apply_reviews(rows, [{"review_id": "pool-18", "approved": True,
                                 "labels": ["退换货"]}])[0]["labels"] == ["退换货"]


@pytest.mark.asyncio
async def test_simulation_batches_until_requested_count():
    from scripts.ch10.build_corpus import simulate_class

    class FakeSimulator:
        def __init__(self):
            self.calls = 0

        def with_structured_output(self, schema, *, method):
            self.schema = schema
            return self

        async def ainvoke(self, prompt):
            self.calls += 1
            count = 20 if self.calls == 1 else 5
            return self.schema(items=[{"text": f"退货问题{self.calls}-{n}", "labels": ["退换货"]}
                                      for n in range(count)])

    model = FakeSimulator()
    rows = await simulate_class(model, "退换货", 25)
    assert len(rows) == 25 and model.calls == 2


@pytest.mark.asyncio
async def test_corpus_does_not_silently_promote_failed_prelabel(monkeypatch):
    from app.config import get_settings
    from scripts.ch10.build_corpus import build_corpus
    from scripts.ch10.prelabel import LabelResult
    from scripts.ch10.validate_golden import GoldenReport

    monkeypatch.setenv("CHAT_BASE_URL", "http://127.0.0.1:9/v1")
    get_settings.cache_clear()

    async def prelabel(text):
        return LabelResult(status="failed", labels=(), reason="model unavailable")

    result = await build_corpus([{"id": 1, "text": "猫粮怎么买"}],
                                prelabel=prelabel, golden_report=GoldenReport(10, 9, 0.9, True),
                                target_per_class=1)
    assert result.status == "partial" and result.labeled == ()
    assert result.failures == (1,)
    assert result.failed_rows[0]["id"] == 1
    get_settings.cache_clear()
