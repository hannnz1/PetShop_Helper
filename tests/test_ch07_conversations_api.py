"""Owner-scoped, paged conversation and original-message reads."""

import asyncio
import time

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db import repository
from app.main import create_app


def _client():
    # These routes only read MySQL and do not need Graph startup or an upstream model.
    return TestClient(create_app())


def test_owner_list_orders_pages_and_previews_without_losing_empty_conversations(
    _test_engine, db_session_factory, db_clean,
):
    async def seed():
        old = await repository.create_conversation("alice")
        empty = await repository.create_conversation("alice")
        latest = await repository.create_conversation("alice")
        foreign = await repository.create_conversation("bob")
        await repository.append_message(old, "tool", "private tool output")
        await repository.append_message(old, "user", "first question")
        await repository.append_message(old, "assistant", "answer")
        marker = await repository.last_message_id(old)
        await repository.advance_layer1(old, marker)
        await repository.commit_summary_segment(old, 0, marker - 1, marker, "fact")
        await repository.append_message(old, "user", "second question")
        await repository.append_message(foreign, "user", "bob secret")
        await repository.append_message(latest, "user", "latest question")
        async with _test_engine.begin() as conn:
            await conn.execute(text("UPDATE conversations SET updated_at='2026-01-01 00:00:00' WHERE id=:id"), {"id": old})
            await conn.execute(text("UPDATE conversations SET updated_at='2026-01-02 00:00:00' WHERE id IN (:a,:b)"), {"a": empty, "b": latest})
        return old, empty, latest

    old, empty, latest = asyncio.run(seed())
    client = _client()
    all_ids = []
    before = None
    while True:
        params = {"user_id": "alice", "limit": 1}
        if before is not None:
            params["before"] = before
        response = client.get("/api/conversations", params=params)
        assert response.status_code == 200
        rows = response.json()["items"]
        all_ids.extend(row["id"] for row in rows)
        before = response.json()["next_cursor"]
        if before is None:
            break
    assert all_ids == [latest, empty, old]
    listed = client.get("/api/conversations", params={"user_id": "alice", "limit": 3}).json()["items"]
    assert [(row["id"], row["preview"], row["has_summary"]) for row in listed] == [
        (latest, "latest question", False), (empty, "", False), (old, "first question", True),
    ]
    assert all(row["updated_at"] for row in listed)
    assert client.get("/api/conversations", params={"user_id": "nobody"}).json()["items"] == []


def test_message_pages_return_visible_originals_and_protect_ownership(
    db_session_factory, db_clean,
):
    async def seed():
        own = await repository.create_conversation("alice")
        foreign = await repository.create_conversation("bob")
        ids = [await repository.append_message(own, "user", "original question")]
        await repository.append_message(own, "tool", "hidden legacy tool", tool_call_id="legacy")
        ids.append(await repository.append_message(own, "assistant", "original answer"))
        ids.append(await repository.append_message(own, "user", "follow-up"))
        await repository.append_message(foreign, "user", "bob secret")
        return own, foreign, ids

    own, foreign, ids = asyncio.run(seed())
    client = _client()
    first = client.get(f"/api/conversations/{own}/messages", params={"user_id": "alice", "limit": 2})
    assert first.status_code == 200
    assert [(m["id"], m["role"], m["content"]) for m in first.json()["items"]] == [
        (ids[0], "user", "original question"), (ids[1], "assistant", "original answer"),
    ]
    after = first.json()["next_cursor"]
    second = client.get(f"/api/conversations/{own}/messages", params={"user_id": "alice", "limit": 2, "after": after})
    assert second.status_code == 200
    assert [(m["id"], m["content"]) for m in second.json()["items"]] == [(ids[2], "follow-up")]
    assert second.json()["next_cursor"] is None
    for conversation_id in (foreign, 999999999):
        response = client.get(f"/api/conversations/{conversation_id}/messages", params={"user_id": "alice"})
        assert response.status_code == 404


def test_completed_turn_moves_older_conversation_to_top_without_losing_id_tiebreak(
    db_session_factory, db_clean,
):
    async def seed():
        older = await repository.create_conversation("alice")
        newer = await repository.create_conversation("alice")
        return older, newer

    older, newer = asyncio.run(seed())
    client = _client()
    initial = client.get("/api/conversations", params={"user_id": "alice"}).json()["items"]
    assert [row["id"] for row in initial] == [newer, older]
    # The schema stores whole seconds. Cross a second boundary to distinguish
    # the parent-row touch from the deterministic ID tie-break tested above.
    time.sleep(1.1)
    asyncio.run(repository.append_turn_messages(older, "late question", [], "late reply"))
    response = client.get("/api/conversations", params={"user_id": "alice", "limit": 1})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["items"]] == [older]
    next_page = client.get("/api/conversations", params={
        "user_id": "alice", "limit": 1, "before": response.json()["next_cursor"],
    })
    assert [row["id"] for row in next_page.json()["items"]] == [newer]


def test_invalid_page_sizes_and_missing_user_are_rejected(db_session_factory, db_clean):
    own = asyncio.run(repository.create_conversation("alice"))
    client = _client()
    for path in ("/api/conversations", f"/api/conversations/{own}/messages"):
        assert client.get(path).status_code == 422
        for size in (0, -1, 51):
            assert client.get(path, params={"user_id": "alice", "limit": size}).status_code == 422
