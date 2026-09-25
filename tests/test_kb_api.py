"""KB preview and ingestion API contracts without live database/model calls."""

from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import kb
from app.db import repository


@pytest.fixture(autouse=True)
def offline_lifecycle_lock(monkeypatch):
    @asynccontextmanager
    async def unlocked():
        yield

    monkeypatch.setattr(repository, "knowledge_lifecycle_lock", unlocked)


def _client():
    app = FastAPI()
    app.include_router(kb.router)
    return TestClient(app)


def test_preview_is_dry_run_and_shows_marked_chunks(monkeypatch):
    async def forbidden(_chunks):
        raise AssertionError("preview must not write")

    monkeypatch.setattr(kb.dualwrite, "write_pending", forbidden)
    request = {
        "content_type": "policy",
        "markdown": "# 售后\n\n## 运费\n\n<!-- key-clause -->\n\n满99包邮。",
    }
    with _client() as client:
        response = client.post("/api/kb/preview", json=request)
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["chunks"][0]["is_key_clause"] == 1


def test_preview_marks_existing_question_answer_pair_without_writing(monkeypatch):
    async def existing():
        return [("运费怎么算？", "满99元包邮。")]

    async def forbidden(_chunks):
        raise AssertionError("preview must not write")

    monkeypatch.setattr(kb.repository, "list_chunk_pairs", existing)
    monkeypatch.setattr(kb.dualwrite, "write_manual_report", forbidden, raising=False)
    with _client() as client:
        response = client.post("/api/kb/preview", json={
            "content_type": "faq", "markdown": "# 运费怎么算\n\n满99元包邮。",
        })
    assert response.status_code == 200
    assert response.json()["chunks"][0]["duplicate"] is True


def test_preview_keeps_chunks_when_duplicate_lookup_is_unavailable(monkeypatch):
    async def offline():
        raise ConnectionError("mysql unavailable")

    monkeypatch.setattr(kb.repository, "list_chunk_pairs", offline)
    with _client() as client:
        response = client.post("/api/kb/preview", json={
            "content_type": "faq", "markdown": "# 运费怎么算\n\n满99元包邮。",
        })
    assert response.status_code == 200
    assert response.json()["chunks"][0]["duplicate"] is None


def test_ingest_reuses_identical_source_chunks(monkeypatch):
    calls = []

    async def fake_write(chunks):
        calls.append(chunks)
        inserted = len(chunks) if len(calls) == 1 else 0
        return [101 for _ in chunks], inserted

    monkeypatch.setattr(kb.dualwrite, "write_manual_report", fake_write, raising=False)
    request = {"content_type": "policy", "markdown": "# 售后\n\n## 运费\n\n满99包邮。"}
    with _client() as client:
        first = client.post("/api/kb/ingest", json=request)
        second = client.post("/api/kb/ingest", json=request)
    assert first.status_code == second.status_code == 200
    assert len(calls) == 2
    assert calls[0][0].section_path == calls[1][0].section_path
    assert first.json()["ids"] == second.json()["ids"] == [101]
    assert first.json()["inserted"] == 1
    assert second.json()["inserted"] == 0
    assert second.json()["skipped"] == 1


def test_ingest_preserves_two_table_chunks_with_same_heading(monkeypatch):
    async def fake_report(chunks):
        table = [chunk for chunk in chunks if chunk.questions == "常见问题处理时限"]
        assert len(table) == 2
        assert table[0].answer != table[1].answer
        return list(range(41, 41 + len(chunks))), len(chunks)

    monkeypatch.setattr(kb.dualwrite, "write_pending_report", fake_report)
    filename = "after-sales-manual.md"
    with _client() as client:
        response = client.post("/api/kb/ingest", json={"filename": filename})
    # The file includes other sections too; both rows of its large table must survive.
    assert response.status_code == 200
    assert response.json()["inserted"] >= 2


def test_manual_fingerprint_uses_question_and_answer():
    first = repository.manual_pair_key("运费 怎么算？", "满 99 元包邮。")
    same = repository.manual_pair_key("运费怎么算", "满99元包邮")
    different_answer = repository.manual_pair_key("运费怎么算", "未满收10元运费")
    assert first == same
    assert first != different_answer
    assert repository.manual_pair_key("费率", "每件 1.5 元") != repository.manual_pair_key(
        "费率", "每件 15 元"
    )


def test_preview_rejects_empty_bad_type_and_path():
    with _client() as client:
        assert client.post("/api/kb/preview", json={"content_type": "policy", "markdown": ""}).status_code == 400
        assert client.post("/api/kb/preview", json={"content_type": "unknown", "markdown": "x"}).status_code == 400
        assert client.post("/api/kb/preview", json={"filename": "../product-faq.md"}).status_code == 400


def test_search_returns_dense_hits(monkeypatch):
    async def fake_search(query, top_k=None, min_score=None):
        assert (query, top_k, min_score) == ("邮费是多少", 2, 0.5)
        return [{"id": 7, "score": 0.8, "question": "运费怎么算", "answer": "满99包邮"}]

    monkeypatch.setattr(kb.retrieval, "search_knowledge", fake_search)
    with _client() as client:
        response = client.post("/api/kb/search", json={
            "query": "邮费是多少", "top_k": 2, "min_score": 0.5,
        })
    assert response.status_code == 200
    assert response.json() == {"route": "dense", "hits": [{
        "id": 7, "score": 0.8, "question": "运费怎么算", "answer": "满99包邮",
    }]}


def test_search_rejects_invalid_top_k_before_embedding():
    with _client() as client:
        response = client.post("/api/kb/search", json={"query": "运费", "top_k": 0})
    assert response.status_code == 422


def test_search_reports_unavailable_without_exposing_upstream_error(monkeypatch):
    async def unavailable(_query, top_k=None, min_score=None):
        raise ConnectionError("private upstream endpoint")

    monkeypatch.setattr(kb.retrieval, "search_knowledge", unavailable)
    with _client() as client:
        response = client.post("/api/kb/search", json={"query": "邮费是多少"})
    assert response.status_code == 503
    assert "private upstream endpoint" not in response.text


def test_vectorize_closes_its_client(monkeypatch):
    class FakeClient:
        closed = False

        def close(self):
            self.closed = True

    fake_client = FakeClient()
    monkeypatch.setattr(kb.milvus_client, "get_client", lambda: fake_client)
    monkeypatch.setattr(kb.milvus_client, "ensure_collection", lambda _client: None)

    async def fake_vectorize(_client):
        return 3

    monkeypatch.setattr(kb.dualwrite, "vectorize_pending", fake_vectorize)
    with _client() as client:
        response = client.post("/api/kb/vectorize")
    assert response.status_code == 200
    assert response.json() == {"vectorized": 3}
    assert fake_client.closed


def test_ingest_vectorizes_after_persisting_and_reuses_existing_ids(monkeypatch):
    events = []

    async def fake_write(chunks):
        events.append("mysql")
        return [9 for _ in chunks]

    class FakeClient:
        def close(self):
            events.append("close")

    async def fake_vectorize(_client):
        events.append("milvus")
        return 1

    async def fake_report(chunks):
        return await fake_write(chunks), len(chunks)

    monkeypatch.setattr(kb.dualwrite, "write_manual_report", fake_report, raising=False)
    monkeypatch.setattr(kb.milvus_client, "get_client", FakeClient)
    monkeypatch.setattr(kb.milvus_client, "ensure_collection", lambda _client: None)
    monkeypatch.setattr(kb.dualwrite, "vectorize_pending", fake_vectorize)
    with _client() as client:
        response = client.post("/api/kb/ingest", json={
            "content_type": "faq", "markdown": "# 运费怎么算\n\n满99包邮。",
            "vectorize": True,
        })
    assert response.status_code == 200
    assert response.json()["ids"] == [9]
    assert events == ["mysql", "milvus", "close"]


def test_ingest_vectorize_failure_keeps_pending_with_retry_message(monkeypatch):
    async def fake_write(_chunks):
        return [9]

    class FakeClient:
        def has_collection(self, _name):
            return True

        def close(self):
            pass

    async def broken_vectorize(_client):
        raise RuntimeError("sensitive upstream failure")

    async def fake_report(chunks):
        return await fake_write(chunks), len(chunks)

    monkeypatch.setattr(kb.dualwrite, "write_manual_report", fake_report, raising=False)
    monkeypatch.setattr(kb.milvus_client, "get_client", FakeClient)
    monkeypatch.setattr(kb.milvus_client, "ensure_collection", lambda _client: None)
    monkeypatch.setattr(kb.dualwrite, "vectorize_pending", broken_vectorize)
    with _client() as client:
        response = client.post("/api/kb/ingest", json={
            "content_type": "faq", "markdown": "# 运费怎么算\n\n满99包邮。",
            "vectorize": True,
        })
    assert response.status_code == 502
    assert "已入库" in response.json()["detail"]
    assert "补跑" in response.json()["detail"]
    assert "sensitive upstream" not in response.text


def test_overview_keeps_sources_when_databases_unavailable(monkeypatch):
    async def broken_stats():
        raise ConnectionError("mysql offline")

    def broken_milvus():
        raise ConnectionError("milvus offline")

    monkeypatch.setattr(kb.repository, "knowledge_stats", broken_stats, raising=False)
    monkeypatch.setattr(kb.repository, "staging_stats", broken_stats, raising=False)
    monkeypatch.setattr(kb.milvus_client, "get_client", broken_milvus)
    with _client() as client:
        response = client.get("/api/kb/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["mysql"]["available"] is False
    assert data["milvus"]["available"] is False
    assert data["consistent"] is None
    assert {item["filename"] for item in data["sources"]} == set(kb.sources.SOURCE_TYPES)


def test_overview_reports_consistency_from_independent_counts(monkeypatch):
    async def stats():
        return {"total": 2, "pending": 1, "done": 1, "key_clauses": 0}

    async def staging():
        return {"extracted": 0, "kept": 0, "discarded": 0, "batches": 0}

    async def recent():
        return []

    class FakeClient:
        def has_collection(self, _name):
            return True

        def close(self):
            pass

    monkeypatch.setattr(kb.repository, "knowledge_stats", stats, raising=False)
    monkeypatch.setattr(kb.repository, "staging_stats", staging, raising=False)
    monkeypatch.setattr(kb.repository, "list_recent_chunks", recent, raising=False)
    monkeypatch.setattr(kb.milvus_client, "get_client", FakeClient)
    monkeypatch.setattr(kb.milvus_client, "ensure_collection", lambda _client: None)
    monkeypatch.setattr(kb.milvus_client, "count", lambda _client: 1)
    with _client() as client:
        response = client.get("/api/kb/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["mysql"]["stats"]["pending"] == 1
    assert data["milvus"]["count"] == 1
    assert data["consistent"] is True


def test_staging_lists_rows_by_status(monkeypatch):
    from types import SimpleNamespace

    async def fake_rows(status):
        if status == "kept":
            return [SimpleNamespace(
                id=4, batch_no="mine-abc", source_ref="conv:2",
                question="如何包邮？", answer="满99元包邮。", status="kept",
            )]
        return []

    monkeypatch.setattr(kb.repository, "list_staging_by_status", fake_rows)
    with _client() as client:
        response = client.get("/api/kb/staging")
    assert response.status_code == 200
    assert response.json()["rows"] == [{
        "id": 4, "batch_no": "mine-abc", "source_ref": "conv:2",
        "question": "如何包邮？", "answer": "满99元包邮。", "status": "kept",
    }]


def test_overview_does_not_create_milvus_collection(monkeypatch):
    class FakeClient:
        def has_collection(self, name):
            assert name == kb.milvus_client.COLLECTION
            return False

        def close(self):
            pass

    monkeypatch.setattr(kb.milvus_client, "get_client", FakeClient)
    monkeypatch.setattr(kb.milvus_client, "ensure_collection", lambda _client: (_ for _ in ()).throw(
        AssertionError("read-only overview must not create collection")
    ))
    with _client() as client:
        response = client.get("/api/kb/overview")
    assert response.status_code == 200
    assert response.json()["milvus"] == {"available": True, "count": 0}
