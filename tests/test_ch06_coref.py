"""Short-history reference resolution without modifying the audited user text."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda


@pytest.mark.asyncio
async def test_resolve_returns_rewrite_and_falls_back_on_blank_or_failure():
    from app.core.coref import resolve

    rewrite = RunnableLambda(lambda _: AIMessage(content="订单1001能退吗"))
    blank = RunnableLambda(lambda _: AIMessage(content="  "))

    def fail(_):
        raise TimeoutError("upstream")

    broken = RunnableLambda(fail)
    assert await resolve("那它能退吗", "用户：订单1001在哪", rewrite) == "订单1001能退吗"
    assert await resolve("那它能退吗", "用户：订单1001在哪", blank) == "那它能退吗"
    assert await resolve("那它能退吗", "用户：订单1001在哪", broken) == "那它能退吗"


@pytest.mark.asyncio
async def test_coref_over_budget_prompt_or_rewrite_falls_back_without_model_cost():
    from app.core.coref import resolve

    calls = []

    def oversized(_):
        calls.append(True)
        return AIMessage(content="订单1001" * 300)

    model = RunnableLambda(oversized)
    assert await resolve("那它能退吗", "用户：订单1001在哪", model, max_tokens=40) == "那它能退吗"
    assert calls == []
    assert await resolve("那它能退吗", "用户：订单1001在哪", model, max_tokens=500) == "那它能退吗"
    assert calls == [True]


@pytest.mark.asyncio
async def test_complete_question_skips_rewrite_model_call():
    from app.core.coref import resolve

    calls = []

    def unexpected(_):
        calls.append(True)
        return AIMessage(content="编造的结果")

    model = RunnableLambda(unexpected)
    assert await resolve("订单1001能退吗", "用户：订单1001在哪", model) == "订单1001能退吗"
    assert await resolve("那它能退吗", "", model) == "那它能退吗"
    assert calls == []


def test_history_excludes_current_human_and_keeps_six_complete_turns():
    from app.graph.nodes import _history_text

    messages = []
    for index in range(8):
        messages.extend([HumanMessage(content=f"历史问题{index}"), AIMessage(content=f"历史答复{index}")])
    messages.append(HumanMessage(content="那它能退吗"))

    history = _history_text(messages, max_turns=6, max_tokens=10000)
    assert "历史问题1" not in history
    assert "历史问题2" in history and "历史答复7" in history
    assert "那它能退吗" not in history


@pytest.mark.asyncio
async def test_graph_coref_preserves_raw_query_and_classifier_uses_resolved():
    from app.graph import nodes

    seen = []

    def rewrite(prompt):
        system, human = prompt.to_messages()
        seen.append((system.content, human.content))
        return AIMessage(content="订单1001能退吗")

    model = RunnableLambda(rewrite)
    state = {
        "query": "那它能退吗",
        "messages": [HumanMessage(content="订单1001到哪了"),
                     AIMessage(content="请看订单页面"),
                     HumanMessage(content="那它能退吗")],
    }
    rewritten = await nodes.coref(state, SimpleNamespace(context={"model": model}))
    assert state["query"] == "那它能退吗"
    assert rewritten["resolved_query"] == "订单1001能退吗"
    assert "订单1001到哪了" in seen[0][0]
    assert "那它能退吗" not in seen[0][0]

    class Classifier:
        async def classify(self, query):
            return "退款退货" if "1001" in query else "unknown"

    routed = await nodes.classify_intent_node(
        {**state, **rewritten}, SimpleNamespace(context={"classifier": Classifier()}),
    )
    assert routed["intent"] == "退款退货"
