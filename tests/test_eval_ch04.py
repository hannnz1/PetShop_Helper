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
async def test_unfaithful_case_saves_exact_evidence_order_without_breaking_report(monkeypatch):
    saved = []

    async def upsert(eval_id, **fields):
        saved.append((eval_id, fields))
        return 1, False

    monkeypatch.setattr(eval_ch04.repository, "upsert_faith_case", upsert)
    hits = [{"id": 42, "section_path": "运费", "question": "邮费", "answer": "满99包邮"},
            {"id": 13, "section_path": "退货", "question": "退货邮费", "answer": "由买家承担"}]
    errors = []
    sample = {"id": "A1", "bucket": "A_policy", "query": "退货运费谁出"}
    await eval_ch04._save_unfaithful_case(sample, "平台承担[2]", "与证据相反", hits, "judge", errors)
    assert errors == []
    assert saved[0][0] == "A1"
    assert saved[0][1]["citations"] == [
        {"n": 1, "chunk_id": 42, "section_path": "运费", "question": "邮费", "answer": "满99包邮"},
        {"n": 2, "chunk_id": 13, "section_path": "退货", "question": "退货邮费", "answer": "由买家承担"},
    ]

    async def broken(*args, **kwargs):
        raise ConnectionError("secret")
    monkeypatch.setattr(eval_ch04.repository, "upsert_faith_case", broken)
    await eval_ch04._save_unfaithful_case(sample, "平台承担[2]", "与证据相反", hits, "judge", errors)
    assert errors == ["ledger A1: ConnectionError"]


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


@pytest.mark.asyncio
async def test_generation_cache_resumes_without_repeating_model_calls(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    sample = {"id": "A1", "bucket": "A_policy", "query": "邮费是多少", "expect_points": ["满99元包邮"],
              "should_refuse": False}
    hits = {
        (strategy, "A1"): [{"id": 7, "question": "运费", "answer": "满99元包邮", "section_path": "运费"}]
        for strategy in eval_ch04.STRATEGIES
    }
    calls = {"answer": 0, "_CoverageJudge": 0, "_FaithJudge": 0}

    class FakeModel(RunnableLambda):
        def __init__(self):
            async def answer(_prompt):
                calls["answer"] += 1
                return AIMessage(content="满99元包邮 [1]")
            super().__init__(answer)

        def with_structured_output(self, schema, method=None):
            async def judge(_prompt):
                calls[schema.__name__] += 1
                if schema is eval_ch04._CoverageJudge:
                    return schema(covered_count=1, reason="covered")
                return schema(faithful=True, reason="supported")
            return RunnableLambda(judge)

    monkeypatch.setattr(eval_ch04, "get_chat_model", lambda **kwargs: FakeModel())
    cache = tmp_path / "generation.jsonl"
    first = await eval_ch04._generation([sample], hits, [], cache_path=cache)
    second = await eval_ch04._generation([sample], hits, [], cache_path=cache)
    assert first == second
    assert calls == {"answer": 4, "_CoverageJudge": 4, "_FaithJudge": 1}
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 4


def test_cache_ignores_torn_multibyte_final_row(tmp_path):
    cache = tmp_path / "cache.jsonl"
    cache.write_bytes(b'{"key":"good","value":1}\n{"key":"bad","value":"\xe4')
    assert eval_ch04._read_cache(cache) == {"good": 1}


def test_generation_cache_scope_tracks_prompt_and_model_adapter(tmp_path, monkeypatch):
    root = tmp_path
    (root / "app/core").mkdir(parents=True)
    (root / "app/core/prompts.py").write_text("first", encoding="utf-8")
    (root / "app/core/llm.py").write_text("adapter", encoding="utf-8")
    monkeypatch.setattr(eval_ch04, "ROOT", root)
    retrieval_cache = root / "retrieval-abc.jsonl"
    first = eval_ch04._generation_cache_path(retrieval_cache)
    (root / "app/core/prompts.py").write_text("changed", encoding="utf-8")
    second = eval_ch04._generation_cache_path(retrieval_cache)
    assert first != second


@pytest.mark.asyncio
async def test_unfaithful_case_is_not_rebilled_or_reopened_on_resume(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    sample = {"id": "A1", "bucket": "A_policy", "query": "邮费是多少", "expect_points": ["满99元包邮"],
              "should_refuse": False}
    hits = {
        (strategy, "A1"): [{"id": 7, "question": "运费", "answer": "满99元包邮", "section_path": "运费"}]
        for strategy in eval_ch04.STRATEGIES
    }
    calls = {"answer": 0, "ledger": 0}

    class FakeModel(RunnableLambda):
        def __init__(self):
            async def answer(_prompt):
                calls["answer"] += 1
                return AIMessage(content="满99元包邮 [1]，且所有订单免费")
            super().__init__(answer)

        def with_structured_output(self, schema, method=None):
            async def judge(_prompt):
                if schema is eval_ch04._CoverageJudge:
                    return schema(covered_count=1, reason="covered")
                return schema(faithful=False, reason="无免费证据")
            return RunnableLambda(judge)

    async def save_case(*args, **kwargs):
        calls["ledger"] += 1

    monkeypatch.setattr(eval_ch04, "get_chat_model", lambda **kwargs: FakeModel())
    monkeypatch.setattr(eval_ch04.repository, "upsert_faith_case", save_case)
    cache = tmp_path / "generation.jsonl"
    first = await eval_ch04._generation([sample], hits, [], cache_path=cache)
    second = await eval_ch04._generation([sample], hits, [], cache_path=cache)
    assert first == second
    assert calls == {"answer": 4, "ledger": 1}


@pytest.mark.asyncio
async def test_generation_cache_write_failure_keeps_report(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    sample = {"id": "D1", "bucket": "D_absent", "query": "有火星车吗", "should_refuse": True}
    hits = {(strategy, "D1"): [{"question": "商品", "answer": "没有火星车"}]
            for strategy in eval_ch04.STRATEGIES}

    class FakeModel(RunnableLambda):
        def __init__(self):
            async def answer(_prompt):
                return AIMessage(content="暂时没有查到")
            super().__init__(answer)

        def with_structured_output(self, schema, method=None):
            return RunnableLambda(lambda _: None)

    def fail_checkpoint(*args):
        raise OSError("disk full")

    monkeypatch.setattr(eval_ch04, "get_chat_model", lambda **kwargs: FakeModel())
    monkeypatch.setattr(eval_ch04, "_append_cache", fail_checkpoint)
    lines = []
    report = await eval_ch04._generation([sample], hits, lines, cache_path=tmp_path / "cache.jsonl")
    assert len(report["records"]) == 4
    assert report["refusal_rate"] == 1.0
    assert any("GENERATION_CACHE_ERROR" in line for line in lines)


@pytest.mark.asyncio
async def test_generation_concurrency_can_be_raised_for_full_eval(monkeypatch):
    import asyncio
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    samples = [{"id": item, "bucket": "D_absent", "query": item, "should_refuse": True}
               for item in ("D1", "D2")]
    hits = {(strategy, sample["id"]): [{"question": "商品", "answer": "未收录"}]
            for strategy in eval_ch04.STRATEGIES for sample in samples}
    active = 0
    peak = 0

    class FakeModel(RunnableLambda):
        def __init__(self):
            async def answer(_prompt):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.03)
                active -= 1
                return AIMessage(content="暂时没有查到")
            super().__init__(answer)

        def with_structured_output(self, schema, method=None):
            return RunnableLambda(lambda _: None)

    monkeypatch.setattr(eval_ch04, "get_chat_model", lambda **kwargs: FakeModel())
    await eval_ch04._generation(samples, hits, [], concurrency=6)
    assert peak == 6


@pytest.mark.asyncio
async def test_partial_generation_report_is_saved_but_cli_status_fails(tmp_path, monkeypatch):
    sample = {"id": "A1", "bucket": "A_policy", "query": "邮费是多少", "expect_section": ["运费"],
              "expect_points": ["满99元包邮"], "should_refuse": False}
    source = tmp_path / "samples.jsonl"
    source.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(eval_ch04, "REPORT_DIR", tmp_path / "reports")

    async def retrieve(*args, **kwargs):
        return {(strategy, "A1"): [] for strategy in eval_ch04.STRATEGIES}

    async def generation(*args, **kwargs):
        return {"errors": ["answer vector/A1: OpenAIRateLimitError"], "records": [
            {"id": "A1", "strategy": "vector", "error": "answer unavailable"},
        ]}

    monkeypatch.setattr(eval_ch04, "_retrieve_all", retrieve)
    monkeypatch.setattr(eval_ch04, "_generation", generation)
    report = await eval_ch04.main(samples_path=source)
    assert report["meta"]["status"] == "partial"
    assert eval_ch04._report_exit_code(report) != 0
    saved = json.loads((tmp_path / "reports/rag_eval.json").read_text(encoding="utf-8"))
    assert saved["meta"]["status"] == "partial"


@pytest.mark.asyncio
async def test_retrieval_error_prevents_complete_report(tmp_path, monkeypatch):
    sample = {"id": "A1", "bucket": "A_policy", "query": "邮费是多少", "expect_section": ["运费"],
              "expect_points": ["满99元包邮"], "should_refuse": False}
    source = tmp_path / "samples.jsonl"
    source.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(eval_ch04, "REPORT_DIR", tmp_path / "reports")

    async def retrieve(samples, lines, cache_path=None):
        lines.append("RETRIEVAL_ERROR hybrid/A1: TimeoutError")
        return {(strategy, "A1"): [] for strategy in eval_ch04.STRATEGIES}

    async def generation(*args, **kwargs):
        return {"errors": [], "records": [
            {"id": "A1", "strategy": strategy, "answer": "满99元包邮"}
            for strategy in eval_ch04.STRATEGIES
        ], "faithfulness": 1.0}

    monkeypatch.setattr(eval_ch04, "_retrieve_all", retrieve)
    monkeypatch.setattr(eval_ch04, "_generation", generation)
    report = await eval_ch04.main(samples_path=source)
    assert report["meta"]["status"] == "partial"
    assert eval_ch04._report_exit_code(report) != 0
