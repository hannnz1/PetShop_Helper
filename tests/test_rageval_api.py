"""The RAG dashboard serves the exact saved report without recalculating metrics."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import rageval


class _Jobs:
    def status(self, name):
        assert name == "eval-rag"
        return {"name": name, "state": "idle", "heavy": True, "log_tail": ""}


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(rageval, "REPORT_PATH", tmp_path / "rag_eval.json")
    app = FastAPI()
    app.state.jobs = _Jobs()
    app.include_router(rageval.router)
    return TestClient(app)


def test_missing_and_corrupt_report_are_actionable(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        missing = client.get("/api/rag-eval/overview")
        assert missing.status_code == 200
        assert missing.json()["present"] is False
        assert missing.json()["job"]["name"] == "eval-rag"
        (tmp_path / "rag_eval.json").write_text('{"meta":', encoding="utf-8")
        assert client.get("/api/rag-eval/overview").json()["present"] is False


def test_dashboard_preserves_saved_metrics_and_chooses_best(tmp_path, monkeypatch):
    report = {
        "meta": {"question_count": 2},
        "retrieval": {
            "vector": {"A_policy": {"count": 2, "mrr": 0.75, "recall_at_k": 1.0}},
            "hybrid_rerank": {"A_policy": {"count": 2, "mrr": 0.5, "recall_at_k": 1.0}},
        },
        "evidence_coverage": {"vector": {"A_policy": 0.3}},
        "generation": None,
    }
    path = tmp_path / "rag_eval.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/api/rag-eval/overview")
    body = response.json()
    assert body["retrieval"] == report["retrieval"]
    assert body["evidence_coverage"] == report["evidence_coverage"]
    assert body["generation"] is None and body["generation_done"] is False
    assert body["best"] == {"strategy": "vector", "mrr": 0.75}
    assert body["present"] is True


def test_faith_case_api_uses_current_report_and_handles_review_errors(tmp_path, monkeypatch):
    report = {
        "meta": {"question_count": 300},
        "retrieval": {"hybrid_rerank": {"A_policy": {"count": 240}}},
        "evidence_coverage": {},
        "generation": {"refusal_rate": 1.0, "faithfulness_cases": [{"id": "A30"}]},
    }
    (tmp_path / "rag_eval.json").write_text(json.dumps(report), encoding="utf-8")

    async def list_cases(**kwargs):
        assert kwargs == {"status": None, "page": 1, "size": 20}
        return {"items": [], "total": 0, "page": 1, "size": 20, "pages": 0,
                "counts": {"未解决": 1, "已解决": 2, "无需解决": 0}}

    async def status_map(ids):
        assert ids == ["A30"]
        return {"A30": "未解决", "C2": "已解决"}

    async def set_status(case_id, status, resolution):
        if case_id == 404:
            return None
        if not resolution or not resolution.strip():
            raise ValueError("resolution is required for reviewed cases")
        return {"id": case_id, "status": status, "resolution": resolution}

    monkeypatch.setattr(rageval.repository, "list_faith_cases", list_cases)
    monkeypatch.setattr(rageval.repository, "faith_case_status_map", status_map)
    monkeypatch.setattr(rageval.repository, "set_faith_case_status", set_status)
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/api/rag-eval/faith-cases")
        assert response.status_code == 200
        assert response.json()["hallucination"]["judged"] == 1
        assert response.json()["hallucination"]["ledger"]["total"] == 3
        assert client.post("/api/rag-eval/faith-cases/1/status",
                           json={"status": "已解决", "resolution": "   "}).status_code == 400
        assert client.post("/api/rag-eval/faith-cases/404/status",
                           json={"status": "已解决", "resolution": "已核验"}).status_code == 404
        assert client.post("/api/rag-eval/faith-cases/1/status",
                           json={"status": "不存在"}).status_code == 422
