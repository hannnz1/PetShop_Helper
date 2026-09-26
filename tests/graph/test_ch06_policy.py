"""Refund policy retrieval merges evidence and fails closed when unsupported."""

from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda


@pytest.mark.asyncio
async def test_expand_queries_limits_nonempty_results_and_falls_back():
    from app.core.query_understanding import expand_queries

    class Model:
        def with_structured_output(self, schema, method):
            return RunnableLambda(lambda _: schema(queries=["退货规则", " ", "运费规则", "退款时效", "多余查询"]))

    class Broken:
        def with_structured_output(self, schema, method):
            raise TimeoutError("unavailable")

    assert await expand_queries("订单1001能退吗", Model()) == ["退货规则", "运费规则", "退款时效"]
    assert await expand_queries("订单1001能退吗", Broken()) == ["订单1001能退吗"]


@pytest.mark.asyncio
async def test_policy_queries_include_order_status_and_keep_best_chunk(monkeypatch):
    from app.graph import nodes

    seeds = []
    searches = []

    async def expand(seed, model):
        seeds.append(seed)
        return ["政策甲", "政策乙"]

    async def search(query, **kwargs):
        searches.append((query, kwargs))
        if query == "政策甲":
            return [{"id": 7, "question": "退货", "answer": "旧片段", "section_path": "退货规则",
                     "content_type": "policy", "rerank_score": 0.80}]
        return [
            {"id": 7, "question": "退货", "answer": "最佳片段", "section_path": "退货规则",
             "content_type": "policy", "rerank_score": 0.99},
            {"id": 8, "question": "运费", "answer": "运费说明", "section_path": "运费规则",
             "content_type": "policy", "rerank_score": 0.98},
        ]

    async def check(query, evidence):
        assert "最佳片段" in evidence[0]
        assert "旧片段" not in " ".join(evidence)
        return {"useful": True, "reason": ""}

    monkeypatch.setattr(nodes.query_understanding, "expand_queries", expand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", search)
    monkeypatch.setattr(nodes.selfcheck, "check_sufficient", check)
    state = {"query": "那单能退吗", "resolved_query": "订单1001能退吗", "order_data": {"status": "已签收"},
             "order_id": "1001", "conversation_id": 1, "route": "refund"}
    result = await nodes.retrieve_policy(state, SimpleNamespace(context={"model": object()}))
    assert seeds == ["订单1001能退吗 已签收"]
    assert [query for query, _ in searches] == ["政策甲", "政策乙"]
    assert all(options["strategy"] == "hybrid_rerank" and options["bm25_query"] == query
               for query, options in searches)
    assert result["sufficient"] is True
    assert [item["id"] for item in result["citations"]] == [7, 8]
    assert "最佳片段" in result["evidence"]


@pytest.mark.asyncio
async def test_weak_policy_evidence_does_not_pass_gate(monkeypatch):
    from app.graph import nodes

    async def expand(seed, model):
        return [seed]

    async def search(query, **kwargs):
        return [{"id": 9, "question": "不相关", "answer": "其他内容", "section_path": "其他",
                 "content_type": "policy", "rerank_score": 0.99}]

    async def check(query, evidence):
        return {"useful": False, "reason": "政策不适用"}

    recorded = []

    async def record(cid, query, source, reason):
        recorded.append((cid, query, source, reason))

    monkeypatch.setattr(nodes.query_understanding, "expand_queries", expand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", search)
    monkeypatch.setattr(nodes.selfcheck, "check_sufficient", check)
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", record)
    state = {"query": "能退吗", "resolved_query": "订单1001能退吗", "order_data": {"status": "已签收"},
             "order_id": "1001", "conversation_id": 11, "route": "refund"}
    result = await nodes.retrieve_policy(state, SimpleNamespace(context={"model": object()}))
    assert result["sufficient"] is False
    assert result["citations"] == []
    assert nodes.confidence_gate(result) == "fallback"
    assert recorded == [(11, "能退吗", "self_check", "政策不适用")]
