from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.memory import SessionStore, build_context, estimate_tokens, trim_history


def test_store_unknown_session_is_empty_without_creating_it():
    store = SessionStore()
    assert store.get("missing") == []
    assert store._sessions == {}


def test_store_append_replace_and_get_are_isolated():
    store = SessionStore()
    first = HumanMessage("first")
    store.append("a", first, AIMessage("answer"))
    store.append("b", HumanMessage("other"))

    first.content = "changed outside"
    fetched = store.get("a")
    assert fetched[0].content == "first"
    fetched[0].content = "changed after get"
    fetched.append(HumanMessage("extra"))
    assert [m.content for m in store.get("a")] == ["first", "answer"]
    assert [m.content for m in store.get("b")] == ["other"]

    replacement = [HumanMessage("new"), AIMessage("reply")]
    store.replace("a", replacement)
    replacement[0].content = "modified"
    assert [m.content for m in store.get("a")] == ["new", "reply"]


def test_estimate_tokens_counts_cjk_conservatively_without_mutating():
    messages = [HumanMessage("喵" * 20), AIMessage("ok")]
    original = deepcopy(messages)
    assert estimate_tokens(messages) >= 20
    assert messages == original


def test_trim_history_keeps_only_newest_complete_pairs():
    messages = [
        HumanMessage("old"), AIMessage("old answer"),
        HumanMessage("new"), AIMessage("new answer"),
    ]
    original = deepcopy(messages)
    latest_budget = estimate_tokens(messages[-2:])
    assert trim_history(messages, latest_budget) == messages[-2:]
    assert trim_history(messages, estimate_tokens(messages)) == messages
    assert messages == original


def test_trim_history_never_keeps_half_turn_or_old_pair_after_newest_fails():
    messages = [
        HumanMessage("tiny"), AIMessage("yes"),
        HumanMessage("喵" * 100), AIMessage("汪" * 100),
    ]
    assert trim_history(messages, estimate_tokens(messages[:2])) == []
    assert trim_history(messages, 0) == []


@pytest.mark.parametrize("messages", [
    [HumanMessage("orphan")],
    [AIMessage("orphan")],
    [HumanMessage("one"), HumanMessage("two")],
    [SystemMessage("not history"), HumanMessage("one")],
])
def test_trim_history_rejects_incomplete_history(messages):
    with pytest.raises(ValueError):
        trim_history(messages, 1000)


def test_build_context_preserves_system_current_and_whole_turns():
    system = SystemMessage("helpful")
    history = [
        HumanMessage("old"), AIMessage("old reply"),
        HumanMessage("new"), AIMessage("new reply"),
    ]
    current = HumanMessage("today")
    budget = estimate_tokens([system, current]) + estimate_tokens(history[-2:])
    context = build_context(system, history, current, budget)
    assert context == [system, *history[-2:], current]
    assert context[0] is not system
    assert context[-1] is not current


def test_build_context_empty_history_and_exact_budget():
    system = SystemMessage("猫")
    current = HumanMessage("你好")
    assert build_context(system, [], current, estimate_tokens([system, current])) == [system, current]


def test_build_context_rejects_current_over_budget():
    system = SystemMessage("system")
    current = HumanMessage("喵" * 100)
    with pytest.raises(ValueError):
        build_context(system, [], current, estimate_tokens([system, current]) - 1)


def test_build_context_counts_system_toward_budget():
    system = SystemMessage("喵" * 100)
    current = HumanMessage("hi")
    with pytest.raises(ValueError):
        build_context(system, [], current, estimate_tokens([current]))
