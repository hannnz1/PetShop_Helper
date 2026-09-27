"""Chapter 7 graph integration: persisted context, model input, and failure edges."""

import json
import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import text

from app.core.context_budget import ContextBudget, ContextBudgetExceeded
from app.db.repository import ContextSnapshot, SummarySegment, VisibleMessage
from app.graph import nodes


def snapshot() -> ContextSnapshot:
    return ContextSnapshot(7, 4, 6,
                           (VisibleMessage(5, "user", "猫砂订单 1001"),
                            VisibleMessage(6, "assistant", "演示订单已签收")),
                           (SummarySegment(1, 1, 4, "早期买过猫粮，订单 8888"),))


def runtime(model=None):
    return SimpleNamespace(context={"model": model, "classifier": None,
                                    "snapshot": snapshot(),
                                    "budget": ContextBudget(8000, 5600, 2400, 6000)})


@pytest.mark.asyncio
async def test_coref_uses_early_summary_on_following_turn_and_logs_history(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="app.graph.nodes")
    seen = []

    async def resolve(query, history, model):
        seen.append(history)
        return "订单 8888 怎么样" if query == "那单呢" else query

    monkeypatch.setattr(nodes.coref_service, "resolve", resolve)
    first = await nodes.coref({"query": "那单呢", "conversation_id": 7}, runtime(object()))
    second = await nodes.coref({"query": "它呢", "conversation_id": 7}, runtime(object()))
    assert first["resolved_query"] == "订单 8888 怎么样"
    assert len(seen) == 2 and all("早期买过猫粮，订单 8888" in item for item in seen)
    assert "history_ctx" in caplog.text


@pytest.mark.asyncio
async def test_chitchat_path_still_records_history_before_routing(caplog):
    caplog.set_level(logging.INFO, logger="app.graph.nodes")
    result = await nodes.coref({"query": "你好", "conversation_id": 7}, runtime(None))
    assert "早期买过猫粮，订单 8888" in result["history_ctx"]
    assert "history_ctx" in caplog.text


@pytest.mark.asyncio
async def test_each_model_call_logs_exact_messages_and_recalculates_tool_exchange(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="app.graph.nodes")
    calls = []

    class Model:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            calls.append(messages)
            return AIMessage(content="planning")

        async def astream(self, messages):
            calls.append(messages)
            yield AIMessage(content="答复")

    monkeypatch.setattr(nodes, "get_chat_tools", lambda route: [])
    state = {"conversation_id": 7, "query": "查订单", "route": "business",
             "messages": [HumanMessage("查订单")], "steps": 0,
             "evidence": "", "tool_results": []}
    first = await nodes.agent_llm(state, runtime(Model()))
    state["messages"] += first["messages"]
    state["messages"] += [ToolMessage("演示记录显示已签收", tool_call_id="call-1")]
    state["steps"] = 1
    second = await nodes.agent_llm(state, runtime(Model()))
    state["messages"] += second["messages"]
    await nodes.final_answer(state, runtime(Model()))
    assert len(calls) == 3
    assert not any(isinstance(message, ToolMessage) for message in calls[0])
    assert any(isinstance(message, ToolMessage) for message in calls[1])
    logged = [json.loads(record.message.removeprefix("model_ctx "))
              for record in caplog.records if record.message.startswith("model_ctx ")]
    assert len(logged) == 3
    for actual, entry in zip(calls, logged, strict=True):
        assert entry["messages"] == [{"role": message.type, "content": message.content,
                                       "name": message.name,
                                       "tool_calls": getattr(message, "tool_calls", None), "tool_call_id": getattr(message, "tool_call_id", None)}
                                      for message in actual]
        assert entry["message_count"] == len(actual)


@pytest.mark.asyncio
async def test_tool_step_rejects_aggregate_results_over_cap(monkeypatch):
    async def fake_execute(call, conversation_id, specs):
        return SimpleNamespace(tool_message=ToolMessage("x" * 3000, tool_call_id=call["id"]),
                               tool_call_id=call["id"], name="query_order", ok=True)

    monkeypatch.setattr(nodes.engine, "execute_tool_call", fake_execute)
    with pytest.raises(ContextBudgetExceeded):
        await nodes.agent_tools({"route": "business", "conversation_id": 7,
                                 "planned_tool_calls": [{"name": "query_order", "id": "a"},
                                                        {"name": "query_order", "id": "b"}],
                                 "tool_results": []})


@pytest.mark.asyncio
async def test_classifier_receives_same_bounded_history_as_coref():
    received = []

    class Classifier:
        async def classify_with_history(self, query, history):
            received.append((query, history))
            return "闲聊"

    context = runtime(None)
    context.context["classifier"] = Classifier()
    state = {"query": "它呢", "resolved_query": "猫粮呢", "conversation_id": 7,
             "history_ctx": "早期摘要：早期买过猫粮，订单 8888"}
    result = await nodes.classify_intent_node(state, context)
    assert result["route"] == "chitchat"
    assert received == [("猫粮呢", state["history_ctx"])]


@pytest.mark.asyncio
async def test_late_stream_failure_never_emits_completion():
    from app.api.chat import graph_event_stream

    async def events():
        yield "updates", {"log_turn": {"conversation_id": 7}}
        raise RuntimeError("checkpoint failed")

    frames = [frame async for frame in graph_event_stream(events(), "alice")]
    assert not any('"event": "done"' in frame or "[DONE]" in frame for frame in frames)
    assert frames[-1].startswith("event: error")


@pytest.mark.asyncio
async def test_runtime_budget_error_is_explicit_and_has_no_completion():
    from app.api.chat import graph_event_stream

    async def events():
        raise ContextBudgetExceeded("current exchange too large")
        yield

    frames = [frame async for frame in graph_event_stream(events(), "alice")]
    assert len(frames) == 1
    assert "上下文预算不足" in frames[0]


@pytest.mark.asyncio
async def test_refund_order_fact_is_appended_in_verified_context(caplog):
    caplog.set_level(logging.INFO, logger="app.graph.nodes")

    class Model:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            return AIMessage(content="stop")

    state = {"conversation_id": 7, "query": "退这个", "route": "refund",
             "messages": [HumanMessage("退这个")], "steps": 0,
             "order_data": {"order_id": "1001", "status": "已签收", "product": "猫粮"}}
    await nodes.agent_llm(state, runtime(Model()))
    entry = next(json.loads(record.message.removeprefix("model_ctx "))
                 for record in caplog.records if record.message.startswith("model_ctx "))
    assert "1001" not in entry["messages"][0]["content"]
    assert "1001" in entry["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_two_graph_turns_read_summary_and_keep_audit_marker(
    tmp_path, db_session_factory, db_clean, monkeypatch, caplog,
):
    from app.db import repository
    from app.graph.build import build_graph
    from app.graph.runtime import GraphRuntime

    caplog.set_level(logging.INFO, logger="app.graph.nodes")
    seen = []

    class Classifier:
        async def classify(self, query):
            return "闲聊"

    async def resolve(query, history, model):
        seen.append(history)
        return "订单 8888 呢"

    monkeypatch.setattr(nodes.coref_service, "resolve", resolve)
    async with GraphRuntime(tmp_path / "context.sqlite", build_graph, classifier=Classifier()) as graph:
        first = await graph.ainvoke_turn("alice", "你好", None, model=object())
        cid = first["conversation_id"]
        marker = first["trace"]["audit_message_id"]
        async with db_session_factory.begin() as session:
            await session.execute(text("""
                INSERT INTO conversation_summaries
                  (conversation_id, seq, from_msg_id, upto_msg_id, content)
                VALUES (:cid, 1, :first, :upto, '早期聊过订单 8888')
            """), {"cid": cid, "first": marker - 1, "upto": marker})
            await session.execute(text("""
                UPDATE conversations SET summary_upto_msg_id=:marker,
                  layer1_from_msg_id=:marker WHERE id=:cid
            """), {"cid": cid, "marker": marker})
        second = await graph.ainvoke_turn("alice", "它呢", cid, model=object())
        assert second["trace"]["audit_message_id"] == await repository.last_message_id(cid)
    assert seen == ["", "早期摘要：早期聊过订单 8888"]
    assert any("history_ctx" in record.message and "订单 8888" in record.message
               for record in caplog.records)


def test_startup_budget_uses_rendered_prompt_and_tool_schemas():
    from app.config import Settings

    good = Settings(_env_file=None, chat_model="offline-test", chat_base_url="http://127.0.0.1:9/v1",
                    chat_api_key="offline-test", token_budget=18000,
                    model_context_window=18000, max_agent_steps=3)
    nodes.validate_startup_budget(good)
    bad = good.model_copy(update={"token_budget": 3000, "model_context_window": 3000})
    with pytest.raises(ContextBudgetExceeded):
        nodes.validate_startup_budget(bad)


def test_tiny_legacy_cap_is_rejected_at_app_startup(tmp_path):
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.main import create_app

    settings = Settings(_env_file=None, chat_model="offline-test", chat_base_url="http://127.0.0.1:9/v1",
                        chat_api_key="offline-test", token_budget=500,
                        graph_checkpoint_path=str(tmp_path / "graph.sqlite"))
    with pytest.raises(ContextBudgetExceeded), TestClient(create_app(settings=settings, model=object())):
        pass


@pytest.mark.asyncio
async def test_model_call_uses_runtime_settings_for_budget(monkeypatch):
    from app.config import Settings

    large = Settings(_env_file=None, chat_model="offline-test", chat_base_url="http://127.0.0.1:9/v1",
                     chat_api_key="offline-test", token_budget=32768)
    small = large.model_copy(update={"token_budget": 500, "model_context_window": 500})
    monkeypatch.setattr(nodes, "get_settings", lambda: large)

    class Model:
        def bind_tools(self, tools):
            raise AssertionError("invalid budget must fail before model binding")

    context = SimpleNamespace(context={"model": Model(), "snapshot": snapshot(), "settings": small})
    with pytest.raises(ContextBudgetExceeded):
        await nodes.agent_llm({"conversation_id": 7, "query": "查订单", "route": "business",
                               "messages": [HumanMessage("查订单")], "steps": 0}, context)


@pytest.mark.asyncio
async def test_model_context_estimate_uses_runtime_cjk_calibration(monkeypatch):
    from app.config import Settings
    from app.core import context_layers, memory
    from app.core.memory import estimate_tokens

    default = Settings(_env_file=None, chat_model="offline-test", chat_base_url="http://127.0.0.1:9/v1",
                       chat_api_key="offline-test", cjk_chars_per_token=4, token_budget=32768)
    custom = default.model_copy(update={"cjk_chars_per_token": 1})
    monkeypatch.setattr(nodes, "get_settings", lambda: default)
    monkeypatch.setattr(context_layers, "get_settings", lambda: default)
    monkeypatch.setattr(memory, "get_settings", lambda: default)

    class Model:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.messages = messages
            return AIMessage(content="stop")

    model = Model()
    context = SimpleNamespace(context={"model": model, "snapshot": snapshot(), "settings": custom})
    result = await nodes.agent_llm({"conversation_id": 7, "query": "查订单", "route": "business",
                                    "messages": [HumanMessage("查订单")], "steps": 0}, context)
    assert result["model_ctx"]["token_estimate"] == estimate_tokens(
        model.messages, chars_per_token=custom.cjk_chars_per_token)


@pytest.mark.asyncio
async def test_compiled_graph_uses_injected_step_limit_not_global(monkeypatch):
    from app.config import Settings
    from app.graph import build
    from app.graph.state import new_turn

    base = Settings(_env_file=None, chat_model="offline-test", chat_base_url="http://127.0.0.1:9/v1",
                    chat_api_key="offline-test", token_budget=32768, max_agent_steps=6)
    injected = base.model_copy(update={"max_agent_steps": 3})
    monkeypatch.setattr(build, "get_settings", lambda: base)

    async def no_log(*args, **kwargs):
        return 1

    monkeypatch.setattr(nodes.repository, "append_turn_messages", no_log)

    class Classifier:
        async def classify(self, query):
            return "订单"

    class Model:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            return AIMessage(content="", tool_calls=[
                {"name": "query_order", "args": {"order_id": "1001"}, "id": f"call-{self.calls}"}])

    model = Model()
    result = await build.build_graph().ainvoke(
        new_turn("alice", 7, "一直查订单 1001"),
        context={"model": model, "classifier": Classifier(),
                 "snapshot": ContextSnapshot(7, 0, 0, (), ()), "settings": injected},
    )
    assert result["steps"] == model.calls == 3
    assert len(result["tool_results"]) == 2
    assert "暂时" in result["answer"]


@pytest.mark.asyncio
async def test_context_log_writes_local_file_without_propagating_raw_text(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path = tmp_path / "log" / "app.log"
    handler = nodes.open_context_log(path)
    try:
        await nodes.coref({"query": "你好", "conversation_id": 7}, runtime(None))
    finally:
        nodes.close_context_log(handler)
    assert "history_ctx" in path.read_text(encoding="utf-8")
    assert "早期买过猫粮" not in caplog.text

@pytest.mark.asyncio
async def test_usage_logs_correlate_estimate_and_stream_usage(monkeypatch, caplog):
    from langchain_core.messages import AIMessageChunk
    caplog.set_level(logging.INFO, logger='app.graph.nodes')
    monkeypatch.setattr(nodes, 'get_chat_tools', lambda route: [])
    usage = {'input_tokens': 100, 'output_tokens': 3, 'total_tokens': 103}

    class Model:
        def bind_tools(self, tools):
            return self
        async def ainvoke(self, messages):
            return AIMessage(content='planning', usage_metadata=usage)
        async def astream(self, messages):
            yield AIMessageChunk(content='答复')
            yield AIMessageChunk(content='', usage_metadata=usage)

    state = {'conversation_id': 7, 'query': '查询', 'route': 'business',
             'messages': [HumanMessage('查询')], 'steps': 0, 'tokens_used': 5}
    first = await nodes.agent_llm(state, runtime(Model()))
    final = await nodes.final_answer({**state, 'tokens_used': first['tokens_used']}, runtime(Model()))
    assert final['tokens_used'] == 211
    logged = [json.loads(r.message.removeprefix('model_usage ')) for r in caplog.records
              if r.message.startswith('model_usage ')]
    assert len(logged) == 2
    contexts = [json.loads(r.message.removeprefix('model_ctx ')) for r in caplog.records
                if r.message.startswith('model_ctx ')]
    assert [row['call_id'] for row in logged] == [row['call_id'] for row in contexts]
    assert len({row['call_id'] for row in logged}) == 2
    for entry in logged:
        assert entry['conversation_id'] == 7
        assert entry['usage'] == usage
        assert entry['input_token_delta'] == 100 - entry['estimated_input_tokens']
