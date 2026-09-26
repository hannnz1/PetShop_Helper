"""Stable FAQ tool registration and semantic search behavior."""

from types import SimpleNamespace

from app.tools.business import FaqInput
from app.tools.business import query_faq


async def test_query_faq_tool_keeps_name_description_and_input_schema(monkeypatch):
    monkeypatch.setattr("app.tools.business.get_settings", lambda: SimpleNamespace(milvus_uri="data/milvus_knowledge.db"))
    async def fake_search(keyword):
        assert keyword == "退货政策"
        return [{"id": 7, "score": 0.91, "question": "退货政策", "answer": "7 天无理由退货"}]

    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    assert query_faq.name == "query_faq"
    assert query_faq.args_schema is FaqInput
    assert query_faq.description == "查询常见问题、政策及商品手册/型号规格知识库；型号问题须在 keyword 中保留完整型号。"
    assert await query_faq.ainvoke({"keyword": "退货政策"}) == {
        "hits": [{"question": "退货政策", "answer": "7 天无理由退货"}]
    }


async def test_query_faq_sends_synonym_query_to_semantic_retrieval(monkeypatch):
    monkeypatch.setattr("app.tools.business.get_settings", lambda: SimpleNamespace(milvus_uri="data/milvus_knowledge.db"))
    calls = []

    async def fake_search(keyword):
        calls.append(keyword)
        return [{"id": 8, "score": 0.87, "question": "运费怎么算", "answer": "按地址计算"}]

    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    assert await query_faq.ainvoke({"keyword": "邮费"}) == {
        "hits": [{"question": "运费怎么算", "answer": "按地址计算"}]
    }
    assert calls == ["邮费"]
