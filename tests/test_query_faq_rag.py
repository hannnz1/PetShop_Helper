"""RAG FAQ path is opt-in until the Standalone data migration."""

from types import SimpleNamespace

import pytest

from app.tools import business


@pytest.fixture(autouse=True)
def standalone_settings(monkeypatch):
    monkeypatch.setattr(business, "get_settings", lambda: SimpleNamespace(
        milvus_uri="http://127.0.0.1:19530", rerank_min_score=0.3,
    ))


@pytest.mark.asyncio
async def test_sufficient_evidence_has_aligned_numbered_citations(monkeypatch):
    async def understand(query):
        return {"standard": "邮费多少", "expanded": ["运费"]}

    async def search(query, **kwargs):
        assert query == "邮费多少 运费"
        assert kwargs == {"strategy": "hybrid_rerank", "category": "运费"}
        return [
            {"id": 1, "rerank_score": 0.9, "question": "运费", "answer": "满99元包邮", "section_path": "运费政策", "content_type": "faq", "category": "运费"},
            {"id": 2, "rerank_score": 0.6, "question": "不足99元", "answer": "收10元运费", "section_path": "运费政策", "content_type": "policy", "category": "运费"},
            {"id": 3, "rerank_score": 0.4, "question": "配送", "answer": "具体以平台为准", "section_path": "配送", "content_type": "policy", "category": "运费"},
        ]

    async def check(query, evidence):
        assert query == "邮费多少"
        assert len(evidence) == 3
        return {"useful": True, "reason": "够答"}

    monkeypatch.setattr(business.query_understanding, "understand", understand)
    monkeypatch.setattr(business.retrieval, "search_knowledge", search)
    monkeypatch.setattr(business.selfcheck, "check_sufficient", check)
    result = await business.query_faq.ainvoke({"keyword": "邮费到底多少", "category": "运费"})
    assert result["sufficient"] is True
    assert [item["id"] for item in result["citations"]] == [1, 3, 2]
    assert [item["n"] for item in result["citations"]] == [1, 2, 3]
    assert result["evidence"].splitlines()[2] == "[3] 不足99元: 收10元运费"


@pytest.mark.asyncio
async def test_no_hits_or_low_score_rejects_before_selfcheck(monkeypatch):
    async def understand(query):
        return {"standard": query, "expanded": []}

    async def no_check(query, evidence):
        raise AssertionError("self-check must not run")

    monkeypatch.setattr(business.query_understanding, "understand", understand)
    monkeypatch.setattr(business.selfcheck, "check_sufficient", no_check)
    for hits in ([], [{"id": 1, "rerank_score": 0.2}]):
        async def search(query, **kwargs):
            return hits

        monkeypatch.setattr(business.retrieval, "search_knowledge", search)
        result = await business.query_faq.ainvoke({"keyword": "火星车怎么买"})
        assert result["sufficient"] is False
        assert result["source"] == "retrieval_low_conf"
        assert result["citations"] == []


@pytest.mark.asyncio
async def test_selfcheck_failure_returns_reason_without_evidence(monkeypatch):
    async def understand(query):
        return {"standard": query, "expanded": []}

    async def search(query, **kwargs):
        return [{"id": 1, "rerank_score": 0.8, "question": "运费", "answer": "满99元包邮", "section_path": "运费", "content_type": "faq", "category": "运费"}]

    async def check(query, evidence):
        return {"useful": False, "reason": "证据只有运费，没有型号功能"}

    monkeypatch.setattr(business.query_understanding, "understand", understand)
    monkeypatch.setattr(business.retrieval, "search_knowledge", search)
    monkeypatch.setattr(business.selfcheck, "check_sufficient", check)
    result = await business.query_faq.ainvoke({"keyword": "Pro 型号功能"})
    assert result == {"sufficient": False, "source": "self_check", "reason": "证据只有运费，没有型号功能", "citations": []}
