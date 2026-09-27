"""Chapter 4 evaluation ledger contracts, using labeled offline samples."""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr


def labeled_report(count=2, status="complete", retrieval_error_count=0):
    return {
        "meta": {"question_count": count + 1, "status": status,
                 "retrieval_error_count": retrieval_error_count,
                 "evaluated_at": "2026-09-27T00:00:00+00:00", "judge_model": "glm-5.2"},
        "retrieval": {"vector": {"A_policy": {"count": count, "recall_at_5": .5, "mrr": .25}}},
        "evidence_coverage": {"vector": {"A_policy": .75}},
        "generation": {"records": [
            {"id": "A1", "bucket": "A_policy", "strategy": "vector", "covered": 1},
            {"id": "A2", "bucket": "A_policy", "strategy": "vector", "covered": None},
        ]},
    }


def test_live_preflight_accepts_siliconflow_key_for_rerank_fallback(tmp_path, monkeypatch):
    from scripts.ch09 import eval_trend

    milvus = tmp_path / "milvus.db"
    milvus.touch()
    settings = SimpleNamespace(chat_model="glm-5.2", siliconflow_api_key=SecretStr("key"),
                               rerank_api_key=None, milvus_uri=str(milvus))
    monkeypatch.setattr(eval_trend, "get_settings", lambda: settings)
    assert eval_trend.upstream_unavailable_reason() is None
    settings.siliconflow_api_key = None
    assert "SiliconFlow" in eval_trend.upstream_unavailable_reason()


def test_extract_preserves_denominators_and_missing_metric_is_pending():
    from app.observability.eval_trend import extract_summary

    summary = extract_summary(labeled_report(), "vector")
    assert summary["status"] == "pending_upstream"
    assert summary["metrics"]["recall_at_5:A_policy"] == {"value": .5, "sample_count": 2}
    assert summary["metrics"]["answer_coverage:A_policy"]["status"] == "pending_upstream"
    assert summary["denominators"]["recall_at_5:A_policy"] == 2
    assert extract_summary(labeled_report(status="partial", retrieval_error_count=1), "vector")["metrics"]["recall_at_5:A_policy"]["value"] is None
    assert extract_summary({"meta": {"status": "complete"}, "retrieval": {}}, "vector")["status"] == "pending_upstream"


@pytest.mark.asyncio
async def test_ledger_is_immutable_and_denominator_changes_are_incomparable(db_session_factory, db_clean):
    from app.db.observability import comparable_trend, record_eval_run

    first = labeled_report()
    first["meta"].update(report_path="data/ch09/eval-r1.json", git_sha="abc")
    await record_eval_run("r1", "a" * 64, "vector", first)
    changed = labeled_report(count=3)
    changed["meta"].update(report_path="data/ch09/eval-r2.json", git_sha="def")
    await record_eval_run("r2", "a" * 64, "vector", changed)
    await record_eval_run("r3", "b" * 64, "vector", first)
    await record_eval_run("r1", "a" * 64, "vector", changed)
    rows = await comparable_trend("a" * 64, "vector")
    assert [row["run_id"] for row in rows] == ["r1", "r2"]
    assert [row["comparable"] for row in rows] == [False, False]
    assert rows[0]["report_path"] == "data/ch09/eval-r1.json"
    assert rows[0]["git_sha"] == "abc"
    assert rows[0]["metrics"]["recall_at_5:A_policy"]["sample_count"] == 2
    assert len(await comparable_trend("b" * 64, "vector")) == 1


def test_trend_api_keeps_bearer_gate(monkeypatch):
    from app.api.observability import router
    from app.db import observability

    async def fake_trend(dataset_hash, strategy):
        assert (dataset_hash, strategy) == ("a" * 64, "vector")
        return [{"run_id": "r1", "comparable": True}]

    monkeypatch.setattr(observability, "comparable_trend", fake_trend)
    app = FastAPI()
    app.include_router(router)
    app.state.settings = type("Config", (), {"observability_admin_token": SecretStr("secret")})()
    path = "/api/observability/eval-trend?dataset_hash=" + "a" * 64 + "&strategy=vector"
    with TestClient(app) as client:
        assert client.get(path).status_code == 401
        response = client.get(path, headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json()[0]["run_id"] == "r1"


@pytest.mark.asyncio
async def test_offline_report_import_writes_unique_local_copy_and_ledger(
    tmp_path, monkeypatch, db_session_factory, db_clean,
):
    from scripts.ch09 import eval_trend
    from app.db.observability import comparable_trend

    source = tmp_path / "labeled.json"
    source.write_text(json.dumps(labeled_report()), encoding="utf-8")
    samples = tmp_path / "samples.jsonl"
    samples.write_text('{"id":"A1"}\n', encoding="utf-8")
    monkeypatch.setattr(eval_trend, "REPORT_DIR", tmp_path / "runs")
    monkeypatch.setattr(eval_trend, "SAMPLES", samples)
    first = await eval_trend.run_once(run_id="offline-1", report_file=source)
    again = await eval_trend.run_once(run_id="offline-1", report_file=source)
    assert first["report_path"] == again["report_path"]
    assert (eval_trend.ROOT / first["report_path"]).is_file() or (tmp_path / "runs" / "offline-1.json").is_file()
    rows = await comparable_trend(first["dataset_hash"], "vector")
    assert len(rows) == 1 and rows[0]["status"] == "pending_upstream"
    assert all(metric["value"] is None for metric in rows[0]["metrics"].values())
    assert rows[0]["comparable"] is False


@pytest.mark.asyncio
async def test_unavailable_upstream_writes_no_stale_scores(tmp_path, monkeypatch, db_session_factory, db_clean):
    from scripts.ch09 import eval_trend

    samples = tmp_path / "samples.jsonl"
    samples.write_text('{"id":"A1","bucket":"A_policy"}\n', encoding="utf-8")
    monkeypatch.setattr(eval_trend, "REPORT_DIR", tmp_path / "runs")
    monkeypatch.setattr(eval_trend, "SAMPLES", samples)
    monkeypatch.setattr(eval_trend, "upstream_unavailable_reason", lambda: "Milvus unavailable")
    result = await eval_trend.run_once(run_id="offline-2", live=True)
    assert result["status"] == "pending_upstream"
    report = json.loads((tmp_path / "runs" / "offline-2.json").read_text(encoding="utf-8"))
    assert report["meta"]["unavailable_reason"] == "Milvus unavailable"
    assert report["retrieval"] == {}


@pytest.mark.asyncio
async def test_run_id_length_fits_all_strategy_rows():
    from scripts.ch09.eval_trend import run_once

    with pytest.raises(ValueError, match="1-50"):
        await run_once(run_id="x" * 51)


@pytest.mark.asyncio
async def test_live_evaluator_disables_cache(tmp_path, monkeypatch, db_session_factory, db_clean):
    from scripts.ch09 import eval_trend
    from scripts import eval_ch04

    samples = tmp_path / "samples.jsonl"
    samples.write_text('{"id":"A1"}\n', encoding="utf-8")
    monkeypatch.setattr(eval_trend, "SAMPLES", samples)
    monkeypatch.setattr(eval_trend, "REPORT_DIR", tmp_path / "runs")
    monkeypatch.setattr(eval_trend, "upstream_unavailable_reason", lambda: None)

    async def fake_main(*, use_cache):
        assert use_cache is False
        return labeled_report()

    monkeypatch.setattr(eval_ch04, "main", fake_main)
    await eval_trend.run_once(run_id="fresh", live=True)


@pytest.mark.asyncio
async def test_pending_first_does_not_become_comparison_baseline(db_session_factory, db_clean):
    from app.db.observability import comparable_trend, record_eval_run

    pending = labeled_report(status="partial")
    pending["retrieval"] = {}
    pending["meta"].update(report_path="pending.json", git_sha="abc", evaluated_at="2026-09-27T00:00:00Z")
    complete = labeled_report()
    complete["generation"]["records"][1]["covered"] = 1
    complete["generation"]["answer_coverage"] = {"vector": {"A_policy": .5}}
    complete["meta"].update(report_path="complete.json", git_sha="abc", evaluated_at="2026-09-27T01:00:00Z")
    await record_eval_run("pending", "c" * 64, "vector", pending)
    await record_eval_run("complete", "c" * 64, "vector", complete)
    rows = await comparable_trend("c" * 64, "vector")
    assert [row["comparable"] for row in rows] == [False, True]
    assert rows[0]["comparison_group"] is None
    assert rows[1]["comparison_group"] is not None


@pytest.mark.asyncio
async def test_two_complete_denominators_have_separate_groups(db_session_factory, db_clean):
    from app.db.observability import comparable_trend, record_eval_run

    first = labeled_report()
    first["generation"]["records"][1]["covered"] = 1
    first["generation"]["answer_coverage"] = {"vector": {"A_policy": .5}}
    first["meta"].update(report_path="first.json", git_sha="abc")
    second = labeled_report(count=3)
    second["generation"]["records"].append(
        {"id": "A3", "bucket": "A_policy", "strategy": "vector", "covered": 1}
    )
    second["generation"]["records"][1]["covered"] = 1
    second["generation"]["answer_coverage"] = {"vector": {"A_policy": .5}}
    second["meta"].update(report_path="second.json", git_sha="abc")
    await record_eval_run("complete-1", "d" * 64, "vector", first)
    await record_eval_run("complete-2", "d" * 64, "vector", second)
    rows = await comparable_trend("d" * 64, "vector")
    assert [row["status"] for row in rows] == ["passed", "passed"]
    assert [row["comparable"] for row in rows] == [True, False]
    assert rows[0]["comparison_group"] != rows[1]["comparison_group"]
