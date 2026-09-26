"""Chapter 4 faithfulness-case ledger: durable cases and explicit review."""

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.rageval import router
from app.db import repository


CITATIONS = [{"n": 1, "chunk_id": 42, "section_path": "商品手册 / MH-W40",
              "question": "滤芯多久换", "answer": "每 3 周更换一次"}]


@pytest_asyncio.fixture
async def client(db_session_factory, db_clean):
    app = FastAPI()
    app.include_router(router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http


async def add(eval_id="B1", **changes):
    values = {"bucket": "B_model", "query": "MH-W40 滤芯多久换？",
              "answer": "每周换[1]", "reason": "证据写每 3 周", "citations": CITATIONS}
    values.update(changes)
    return await repository.upsert_faith_case(eval_id, **values)


@pytest.mark.asyncio
async def test_insert_and_repeat_keep_one_row_with_evidence_snapshot(client):
    case_id, reopened = await add()
    assert reopened is False
    again_id, reopened = await add(answer="隔天换[1]", reason="第二次编造")
    assert again_id == case_id and reopened is False
    data = (await client.get("/api/rag-eval/faith-cases")).json()
    assert data["total"] == 1
    assert data["counts"] == {"未解决": 1, "已解决": 0, "无需解决": 0}
    item = data["items"][0]
    assert item["answer"] == "隔天换[1]" and item["seen_count"] == 2
    assert item["citations"] == CITATIONS


@pytest.mark.asyncio
async def test_resolved_case_recurs_and_manual_reopen_clears_resolution(client):
    case_id, _ = await add()
    response = await client.post(f"/api/rag-eval/faith-cases/{case_id}/status",
                                 json={"status": "已解决", "resolution": "补全了型号资料"})
    assert response.status_code == 200
    assert response.json()["resolution"] == "补全了型号资料"
    _, reopened = await add(answer="又答错")
    assert reopened is True
    item = (await client.get("/api/rag-eval/faith-cases")).json()["items"][0]
    assert item["status"] == "未解决" and item["reopened"] is True
    response = await client.post(f"/api/rag-eval/faith-cases/{case_id}/status",
                                 json={"status": "未解决"})
    assert response.json()["resolution"] is None
    assert response.json()["resolved_at"] is None
    assert response.json()["reopened"] is False


@pytest.mark.asyncio
async def test_filter_pages_and_status_validation(client):
    for number in range(6):
        await add(f"A{number}")
    case_id, _ = await add("E1")
    await client.post(f"/api/rag-eval/faith-cases/{case_id}/status",
                      json={"status": "无需解决", "resolution": "裁判误判"})
    first = (await client.get("/api/rag-eval/faith-cases?size=5&page=1")).json()
    second = (await client.get("/api/rag-eval/faith-cases?size=5&page=2")).json()
    assert first["total"] == 7 and first["pages"] == 2
    assert len(first["items"]) == 5 and len(second["items"]) == 2
    assert all(item["status"] == "未解决" for item in first["items"])
    filtered = (await client.get("/api/rag-eval/faith-cases?status=无需解决")).json()
    assert filtered["total"] == 1 and filtered["items"][0]["eval_id"] == "E1"
    assert (await client.get("/api/rag-eval/faith-cases?status=无效")).status_code == 422


@pytest.mark.asyncio
async def test_resolution_required_and_missing_case(client):
    case_id, _ = await add()
    for status in ("已解决", "无需解决"):
        for resolution in (None, "   "):
            body = {"status": status}
            if resolution is not None:
                body["resolution"] = resolution
            response = await client.post(f"/api/rag-eval/faith-cases/{case_id}/status", json=body)
            assert response.status_code == 400
    assert (await client.post(f"/api/rag-eval/faith-cases/{case_id}/status",
                              json={"status": "关闭"})).status_code == 422
    assert (await client.post("/api/rag-eval/faith-cases/99999/status",
                              json={"status": "未解决"})).status_code == 404
    item = (await client.get("/api/rag-eval/faith-cases")).json()["items"][0]
    assert item["status"] == "未解决"
