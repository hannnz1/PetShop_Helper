"""MySQL audit is atomic and stale SQLite checkpoints never resume silently."""

import pytest
from sqlalchemy.exc import DataError

from app.db import repository


class Classifier:
    async def classify(self, query):
        return "闲聊"


@pytest.mark.asyncio
async def test_batch_audit_rolls_back_on_late_answer_error(db_session_factory, db_clean):
    conversation_id = await repository.create_conversation("owner")
    with pytest.raises(DataError):
        await repository.append_turn_messages(conversation_id, "你好", [
            {"content": "x", "tool_call_id": "a" * 65},
        ], "x" * 70000)
    assert await repository.list_messages(conversation_id) == []


@pytest.mark.asyncio
async def test_new_turn_audit_records_only_visible_rows_and_final_marker(db_session_factory, db_clean):
    conversation_id = await repository.create_conversation("owner")
    marker = await repository.append_turn_messages(conversation_id, "查询订单", [
        {"content": "工具结果", "tool_call_id": "old-tool"},
    ], "处理好了")
    rows = await repository.list_messages(conversation_id)
    assert [(row.role, row.content) for row in rows] == [
        ("user", "查询订单"), ("assistant", "处理好了"),
    ]
    assert marker == rows[-1].id == await repository.last_message_id(conversation_id)


@pytest.mark.asyncio
async def test_checkpoint_loss_is_reported_before_new_turn(tmp_path, db_session_factory, db_clean):
    from app.graph.build import build_graph
    from app.graph.runtime import GraphDivergence, GraphRuntime

    first_path = tmp_path / "first.sqlite"
    async with GraphRuntime(first_path, build_graph, classifier=Classifier()) as runtime:
        first = await runtime.ainvoke_turn("owner", "你好", None, model=object())
    conversation_id = first["conversation_id"]
    assert [row.role for row in await repository.list_messages(conversation_id)] == ["user", "assistant"]
    async with GraphRuntime(tmp_path / "lost.sqlite", build_graph, classifier=Classifier()) as runtime:
        with pytest.raises(GraphDivergence):
            await runtime.ainvoke_turn("owner", "还有呢", conversation_id, model=object())
    assert len(await repository.list_messages(conversation_id)) == 2


@pytest.mark.asyncio
async def test_mysql_audit_change_invalidates_old_checkpoint(tmp_path, db_session_factory, db_clean):
    from app.graph.build import build_graph
    from app.graph.runtime import GraphDivergence, GraphRuntime

    async with GraphRuntime(tmp_path / "graph.sqlite", build_graph, classifier=Classifier()) as runtime:
        first = await runtime.ainvoke_turn("owner", "你好", None, model=object())
        conversation_id = first["conversation_id"]
        await repository.append_message(conversation_id, "user", content="外部写入")
        with pytest.raises(GraphDivergence):
            await runtime.ainvoke_turn("owner", "继续", conversation_id, model=object())
