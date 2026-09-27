"""A freshly discovered MCP tool reaches the model and the unified executor."""

from types import SimpleNamespace
import json

from langchain_core.messages import AIMessage, ToolMessage

from app.graph import nodes
from app.tools.engine import ToolRun
from app.tools.registry import ToolSpec


class FakeTool:
    name = "query_delivery_eta"
    description = "查询预计到达时间"
    args_schema = {"type": "object", "properties": {"tracking_no": {"type": "string"}},
                   "required": ["tracking_no"]}


def test_context_budget_uses_the_same_dynamic_tool_schema_as_model_binding():
    tool = FakeTool()
    spec = ToolSpec(tool.name, tool.description, tool.args_schema, tool, "read", "mcp", "logistics")
    rendered = json.loads(nodes._tool_schema_text("business", [spec]))
    assert [item["name"] for item in rendered] == ["query_delivery_eta"]
    assert rendered[0]["schema"] == tool.args_schema


async def test_agent_binds_and_executes_newly_discovered_mcp_tool(monkeypatch):
    discovery_calls = []
    tool = FakeTool()
    spec = ToolSpec(tool.name, tool.description, tool.args_schema, tool, "read", "mcp", "logistics")

    async def discover():
        discovery_calls.append(True)
        return [spec]

    monkeypatch.setattr(nodes.registry, "get_all_specs", discover)
    monkeypatch.setattr(nodes, "_model_input", lambda *_args, **_kw: ([], {"summary_layer2_budget": 100}))
    monkeypatch.setattr(nodes, "_log_model_usage", lambda *_: None)

    class Model:
        def bind_tools(self, tools):
            assert [item.name for item in tools] == ["query_delivery_eta"]
            return self

        async def ainvoke(self, _messages):
            return AIMessage(content="", tool_calls=[
                {"id": "new-call", "name": "query_delivery_eta", "args": {"tracking_no": "SF1"}},
            ])

    runtime = SimpleNamespace(context={"model": Model()})
    planned = await nodes.agent_llm({"route": "business", "steps": 0}, runtime)
    assert planned["planned_tool_calls"][0]["name"] == "query_delivery_eta"

    async def execute(call, conversation_id, specs, **_kwargs):
        assert conversation_id == 7 and call["name"] in specs
        message = ToolMessage(content='{"arrival":"明天"}', tool_call_id=call["id"], name=call["name"])
        return ToolRun(call["id"], call["name"], True, message, "成功")

    monkeypatch.setattr(nodes.engine, "execute_tool_call", execute)
    result = await nodes.agent_tools({
        "route": "business", "conversation_id": 7,
        "planned_tool_calls": planned["planned_tool_calls"],
    }, SimpleNamespace(context={"settings": SimpleNamespace(cjk_chars_per_token=1.0,
                                                               tool_result_max_tokens=1200)}))
    assert result["tool_results"][0]["ok"]
    assert len(discovery_calls) == 2
