"""Offline graph exits never need a paid model for fixed responses."""

import pytest

from app.graph.build import build_graph
from app.graph.state import new_turn


class Classifier:
    def __init__(self, intent):
        self.intent = intent

    async def classify(self, query):
        return self.intent


@pytest.mark.asyncio
@pytest.mark.parametrize("intent,expected", [
    ("投诉", "create_ticket"),
    ("闲聊", None),
], ids=["complaint", "chitchat"])
async def test_fixed_graph_paths_do_not_call_model_or_create_ticket(monkeypatch, intent, expected):
    from app.graph import nodes

    async def no_log(*args, **kwargs):
        return 1

    async def forbidden(*args, **kwargs):
        raise AssertionError("fixed exit must not write ticket")

    monkeypatch.setattr(nodes.repository, "append_message", no_log)
    monkeypatch.setattr(nodes.repository, "create_ticket_only", forbidden)
    result = await build_graph().ainvoke(new_turn("owner", 1, "我要投诉"),
                                        context={"classifier": Classifier(intent), "model": object()})
    assert result["answer"]
    assert result["tool_calls"] == []
    assert [item["type"] for item in result["suggested_actions"]] == (
        ["transfer_human", "create_ticket"] if expected else []
    )
