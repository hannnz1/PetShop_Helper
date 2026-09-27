"""Intent usage report reads only measured tokens from isolated MySQL."""

import json
from datetime import date, datetime
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.db.models import ModelUsageEvent


@pytest.mark.asyncio
async def test_usage_by_day_groups_actual_tokens_and_keeps_unavailable_distinct(db_session_factory, db_clean):
    from app.db.observability import usage_by_day

    values = [
        ("2026-09-25", "refund", "model-a", 10, 4, "available"),
        ("2026-09-25", "refund", "model-a", 0, 0, "available"),
        ("2026-09-25", "refund", "model-a", None, None, "unavailable"),
        ("2026-09-25", "knowledge", "model-b", 5, 2, "available"),
        ("2026-09-26", "refund", "model-b", None, None, "unavailable"),
        ("2026-09-27", "refund", "model-b", 99, 99, "available"),
    ]
    async with db_session_factory.begin() as session:
        session.add_all(ModelUsageEvent(
            run_id=str(uuid4()), conversation_id=101, turn_id=str(uuid4()),
            created_at=datetime.fromisoformat(day + "T12:00:00"),
            intent=intent, model_name=model, input_tokens=input_tokens,
            output_tokens=output_tokens, usage_status=status,
            turn_status="completed", is_estimated=False,
        ) for day, intent, model, input_tokens, output_tokens, status in values)
        session.add(ModelUsageEvent(
            run_id=str(uuid4()), conversation_id=101, turn_id=str(uuid4()),
            created_at=datetime(2026, 9, 25, 12), intent="refund", model_name="model-a",
            input_tokens=999, output_tokens=999, usage_status="available",
            turn_status="completed", is_estimated=True,
        ))

    rows = await usage_by_day(date(2026, 9, 25), date(2026, 9, 26))
    assert rows == [
        {"day": "2026-09-25", "intent": "knowledge", "model_name": "model-b",
         "calls": 1, "available_calls": 1, "unavailable_calls": 0,
         "input_tokens": 5, "output_tokens": 2},
        {"day": "2026-09-25", "intent": "refund", "model_name": "model-a",
         "calls": 3, "available_calls": 2, "unavailable_calls": 1,
         "input_tokens": 10, "output_tokens": 4},
        {"day": "2026-09-26", "intent": "refund", "model_name": "model-b",
         "calls": 1, "available_calls": 0, "unavailable_calls": 1,
         "input_tokens": None, "output_tokens": None},
    ]
    assert "cost" not in json.dumps(rows).lower()


def test_usage_api_rejects_missing_unset_and_wrong_bearer_without_query(monkeypatch):
    from app.api.observability import router
    from app.db import observability as db_observability

    async def forbidden_query(*_):
        raise AssertionError("unauthorized request queried database")

    monkeypatch.setattr(db_observability, "usage_by_day", forbidden_query)
    app = FastAPI()
    app.include_router(router)
    app.state.settings = type("Config", (), {"observability_admin_token": None})()
    with TestClient(app) as client:
        assert client.get("/api/observability/usage?from=2026-09-25&to=2026-09-26",
                          headers={"Authorization": "Bearer configured"}).status_code == 503
        app.state.settings.observability_admin_token = SecretStr("configured")
        for headers in ({}, {"Authorization": "Basic configured"},
                        {"Authorization": "Bearer wrong"},
                        {"Authorization": b"Bearer caf\xe9"}):
            assert client.get("/api/observability/usage?from=2026-09-25&to=2026-09-26",
                              headers=headers).status_code == 401


def test_usage_api_validates_dates_and_returns_only_report(monkeypatch):
    from app.api.observability import router
    from app.db import observability as db_observability

    async def report(start, end):
        assert (start, end) == (date(2026, 9, 25), date(2026, 9, 26))
        return [{"day": "2026-09-25", "intent": "refund", "model_name": "model-a",
                 "calls": 1, "available_calls": 1, "unavailable_calls": 0,
                 "input_tokens": 3, "output_tokens": 2}]

    monkeypatch.setattr(db_observability, "usage_by_day", report)
    app = FastAPI()
    app.include_router(router)
    app.state.settings = type("Config", (), {"observability_admin_token": SecretStr("configured")})()
    headers = {"Authorization": "Bearer configured"}
    with TestClient(app) as client:
        for query in ("from=bad&to=2026-09-26", "from=2026-09-27&to=2026-09-26"):
            assert client.get("/api/observability/usage?" + query, headers=headers).status_code == 422
        response = client.get("/api/observability/usage?from=2026-09-25&to=2026-09-26",
                              headers=headers)
    assert response.status_code == 200
    assert response.json() == [{"day": "2026-09-25", "intent": "refund", "model_name": "model-a",
                                "calls": 1, "available_calls": 1, "unavailable_calls": 0,
                                "input_tokens": 3, "output_tokens": 2}]


def test_cli_prints_json_and_readable_table(monkeypatch, capsys):
    from scripts.ch09 import usage_report

    async def report(start, end):
        assert (start, end) == (date(2026, 9, 25), date(2026, 9, 26))
        return [{"day": "2026-09-25", "intent": "refund", "model_name": "model-a",
                 "calls": 1, "available_calls": 0, "unavailable_calls": 1,
                 "input_tokens": None, "output_tokens": None}]

    monkeypatch.setattr(usage_report, "usage_by_day", report)
    args = ["--from", "2026-09-25", "--to", "2026-09-26"]
    usage_report.main(args + ["--format", "json"])
    assert json.loads(capsys.readouterr().out)[0]["input_tokens"] is None
    usage_report.main(args + ["--format", "table"])
    table = capsys.readouterr().out
    assert "refund" in table and "model-a" in table and "unavailable" in table
    assert "null" not in table
