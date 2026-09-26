"""Deterministic metrics must grade source sections and facts, not model guesses."""

import json

import pytest

from scripts import eval_ch04


def test_rank_and_recall_use_section_path():
    sample = {"expect_section": ["运费怎么算"]}
    hits = [
        {"section_path": "商品与购物 FAQ > 退款", "answer": "满 99 元包邮"},
        {"section_path": "商品与购物 FAQ > 运费怎么算", "answer": "满 99 元包邮"},
    ]
    assert eval_ch04._first_rank(sample, hits) == 2
    assert eval_ch04._first_rank(sample, hits[:1]) is None


def test_evidence_coverage_ignores_whitespace_but_not_missing_facts():
    hits = [{"answer": "满 99 元包邮，未满收取 10 元运费。"}]
    assert eval_ch04._coverage_mech(["满99元包邮", "10元运费"], hits) == 1.0
    assert eval_ch04._coverage_mech(["满99元包邮", "7天送达"], hits) == 0.5
    assert eval_ch04._coverage_mech([], hits) is None


def test_bucket_metrics_exclude_refusal_items():
    samples = [
        {"id": "A1", "bucket": "A_policy", "expect_section": ["运费怎么算"], "expect_points": ["满99元包邮"], "should_refuse": False},
        {"id": "A2", "bucket": "A_policy", "expect_section": ["退货政策是什么"], "expect_points": ["7天"], "should_refuse": False},
        {"id": "D1", "bucket": "D_absent", "expect_section": [], "expect_points": [], "should_refuse": True},
    ]
    hits = {
        ("vector", "A1"): [{"section_path": "退货"}, {"section_path": "运费怎么算", "answer": "满99元包邮"}],
        ("vector", "A2"): [], ("vector", "D1"): [],
    }
    metrics = eval_ch04._deterministic_summary(samples, hits, ["vector"], k=10)
    assert metrics["retrieval"]["vector"]["A_policy"] == {"count": 2, "recall_at_k": 0.5, "mrr": 0.25}
    assert metrics["evidence_coverage"]["vector"]["A_policy"] == 0.5
    assert "D_absent" not in metrics["retrieval"]["vector"]


@pytest.mark.asyncio
async def test_partial_report_survives_generation_skip(tmp_path, monkeypatch):
    sample = {"id": "A1", "bucket": "A_policy", "query": "运费？", "expect_section": ["运费怎么算"],
              "expect_points": ["满99元包邮"], "should_refuse": False}
    source = tmp_path / "samples.jsonl"
    source.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(eval_ch04, "REPORT_DIR", tmp_path / "reports")

    async def retrieve(*args, **kwargs):
        return [{"section_path": "运费怎么算", "answer": "满 99 元包邮"}]

    async def understand(query):
        return {"standard": query, "expanded": []}

    monkeypatch.setattr(eval_ch04.query_understanding, "understand", understand)
    monkeypatch.setattr(eval_ch04.retrieval, "search_knowledge", retrieve)
    report = await eval_ch04.main(skip_generation=True, samples_path=source)
    assert report["generation"] is None
    assert report["retrieval"]["vector"]["A_policy"]["mrr"] == 1.0
    saved = json.loads((tmp_path / "reports/rag_eval.json").read_text(encoding="utf-8"))
    assert saved["evidence_coverage"]["bm25"]["A_policy"] == 1.0
    assert "RETRIEVAL" in (tmp_path / "reports/rag_eval.txt").read_text(encoding="utf-8")
