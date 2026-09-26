"""The public FAQ tool contract when retrieval is backed by vectors."""

from types import SimpleNamespace

from app.tools.business import query_faq


async def test_query_faq_maps_hits_to_contract(monkeypatch):
    monkeypatch.setattr("app.tools.business.get_settings", lambda: SimpleNamespace(milvus_uri="data/milvus_knowledge.db"))
    calls = []

    async def fake_search(keyword):
        calls.append(keyword)
        return [
            {"id": 1, "score": 0.8, "question": "运费怎么算", "answer": "满99包邮"},
            {"id": 2, "score": 0.7, "question": "配送范围", "answer": "全国配送"},
        ]

    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "邮费是多少"})

    assert calls == ["邮费是多少"]
    assert out == {"hits": [
        {"question": "运费怎么算", "answer": "满99包邮"},
        {"question": "配送范围", "answer": "全国配送"},
    ]}


async def test_query_faq_empty_returns_message(monkeypatch):
    monkeypatch.setattr("app.tools.business.get_settings", lambda: SimpleNamespace(milvus_uri="data/milvus_knowledge.db"))
    async def fake_search(keyword):
        assert keyword == "无关问题"
        return []

    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "无关问题"})

    assert out == {"hits": [], "message": "未找到与「无关问题」相关的常见问题"}
