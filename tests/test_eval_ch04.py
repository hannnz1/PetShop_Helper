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
    assert metrics["retrieval"]["vector"]["A_policy"] == {"count": 2, "recall_at_5": 0.5, "mrr": 0.25}
    assert metrics["evidence_coverage"]["vector"]["A_policy"] == 0.5
    assert "D_absent" not in metrics["retrieval"]["vector"]


def test_multi_section_recall_is_partial_and_mrr_averages_evidence_ranks():
    sample = {"id": "E1", "bucket": "E_multi", "expect_section": ["运费", "会员"],
              "expect_sections_all": [["运费"], ["会员权益", "金卡权益"]],
              "expect_points": ["满99", "会员免运费"], "should_refuse": False}
    hits = {("bm25", "E1"): [
        {"section_path": "运费政策", "answer": "满99"},
        {"section_path": "噪声", "answer": "其他"},
        {"section_path": "金卡权益", "answer": "会员免运费"},
    ]}
    out = eval_ch04._deterministic_summary([sample], hits, ["bm25"])
    assert out["retrieval"]["bm25"]["E_multi"] == {
        "count": 1, "recall_at_5": 1.0, "mrr": (1 + 1/3)/2,
    }
    missing = {("bm25", "E1"): hits[("bm25", "E1")][:1]}
    out = eval_ch04._deterministic_summary([sample], missing, ["bm25"])
    assert out["retrieval"]["bm25"]["E_multi"]["recall_at_5"] == 0.5
    assert out["retrieval"]["bm25"]["E_multi"]["mrr"] == 0.5


def test_recall_at_five_does_not_count_rank_six_but_mrr_does():
    sample = {"id": "A1", "bucket": "A_policy", "expect_section": ["目标"],
              "expect_points": ["事实"], "should_refuse": False}
    hits = {("vector", "A1"): [{"section_path": "噪声"}] * 5 +
            [{"section_path": "目标", "answer": "事实"}]}
    out = eval_ch04._deterministic_summary([sample], hits, ["vector"])
    assert out["retrieval"]["vector"]["A_policy"]["recall_at_5"] == 0
    assert out["retrieval"]["vector"]["A_policy"]["mrr"] == 1/6


def test_cache_recovers_after_interrupted_final_line(tmp_path):
    cache = tmp_path / "retrieval.jsonl"
    cache.write_text('{"key":"interrupted"', encoding="utf-8")
    eval_ch04._append_cache(cache, "complete", [{"id": 1}])
    assert eval_ch04._read_cache(cache)["complete"] == [{"id": 1}]


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


@pytest.mark.asyncio
async def test_retrieval_cache_resumes_without_repeating_upstream_calls(tmp_path, monkeypatch):
    sample = {"id": "A1", "query": "满多少包邮"}
    counts = {"rewrite": 0, "search": 0}

    async def understand(query):
        counts["rewrite"] += 1
        return {"standard": query, "expanded": ["运费"]}

    async def search(query, **kwargs):
        counts["search"] += 1
        return [{"id": 7, "section_path": "运费", "answer": "满99元包邮"}]

    monkeypatch.setattr(eval_ch04.query_understanding, "understand", understand)
    monkeypatch.setattr(eval_ch04.retrieval, "search_knowledge", search)
    cache = tmp_path / "retrieval.jsonl"
    first = await eval_ch04._retrieve_all([sample], [], cache_path=cache)
    second = await eval_ch04._retrieve_all([sample], [], cache_path=cache)
    assert first == second
    assert counts == {"rewrite": 1, "search": 4}
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 5


@pytest.mark.asyncio
async def test_cache_retries_only_failed_retrieval_and_ignores_changed_question(tmp_path, monkeypatch):
    sample = {"id": "A1", "query": "满多少包邮"}
    searches = []

    async def understand(query):
        return {"standard": query, "expanded": []}

    async def search(query, **kwargs):
        searches.append(kwargs["strategy"])
        if kwargs["strategy"] == "bm25" and searches.count("bm25") == 1:
            raise RuntimeError("temporary")
        return [{"answer": query}]

    monkeypatch.setattr(eval_ch04.query_understanding, "understand", understand)
    monkeypatch.setattr(eval_ch04.retrieval, "search_knowledge", search)
    cache = tmp_path / "retrieval.jsonl"
    first = await eval_ch04._retrieve_all([sample], [], cache_path=cache)
    assert first[("bm25", "A1")] == []
    await eval_ch04._retrieve_all([sample], [], cache_path=cache)
    assert searches.count("bm25") == 2
    assert len(searches) == 5
    await eval_ch04._retrieve_all([{"id": "A1", "query": "退货邮费"}], [], cache_path=cache)
    assert len(searches) == 9
