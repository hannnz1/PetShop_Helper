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
