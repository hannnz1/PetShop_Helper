"""Chapter 5 workflow nodes, beginning with the mandatory knowledge gate."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.config import get_settings
from app.core.agent import ContextBudgetExceeded, FAQ_REFUSAL
from app.core.intent import safe_classify
from app.core.memory import estimate_tokens
from app.core.prompts import GRAPH_AGENT_SYSTEM, GRAPH_FINAL_SYSTEM
from app.db import repository
from app.graph.routing import route_by_intent
from app.tools.business import query_faq
from app.tools.infra import execute_tool_call
from app.tools.registry import get_chat_tools


async def coref(state: dict) -> dict:
    """Chapter 5 passes the user wording through; Chapter 6 resolves references."""
    return {"query": state["query"]}


async def classify_intent_node(state: dict, runtime: Runtime[dict]) -> dict:
    classifier = runtime.context.get("classifier") if runtime.context else None
    intent = await safe_classify(classifier, state["query"]) if classifier else "unknown"
    return {"intent": intent, "route": route_by_intent(intent)}


def _fit_messages(state: dict, system_text: str) -> list:
    """Keep latest complete prior turns and the entire current tool exchange."""
    messages = state["messages"]
    current_index = max(
        index for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    )
    prior = messages[:current_index]
    current = messages[current_index:]
    system = [SystemMessage(content=system_text)]
    if state.get("evidence"):
        system.append(SystemMessage(content=f"本轮已核验证据:\n{state['evidence']}"))
    budget = get_settings().token_budget
    if estimate_tokens([*system, *current]) > budget:
        raise ContextBudgetExceeded("current graph exchange exceeds context token budget")

    pairs = []
    pending = None
    for message in prior:
        if isinstance(message, HumanMessage):
            pending = message
        elif isinstance(message, AIMessage) and not message.tool_calls and pending is not None:
            pairs.append((pending, message))
            pending = None
    kept = []
    for pair in reversed(pairs):
        candidate = [*pair, *kept]
        if estimate_tokens([*system, *candidate, *current]) > budget:
            break
        kept = candidate
    return [*system, *kept, *current]


async def agent_llm(state: dict, runtime: Runtime[dict]) -> dict:
    tools = get_chat_tools(state["route"])
    model = runtime.context["model"]
    planned = await model.bind_tools(tools).ainvoke(_fit_messages(state, GRAPH_AGENT_SYSTEM))
    calls = list(planned.tool_calls or [])
    usage = planned.usage_metadata or {}
    tokens = int(usage.get("total_tokens") or 0)
    return {
        "messages": [planned], "planned_tool_calls": calls,
        "tool_calls": [*state.get("tool_calls", []), *calls],
        "steps": state.get("steps", 0) + 1,
        "tokens_used": state.get("tokens_used", 0) + tokens,
    }


async def agent_tools(state: dict) -> dict:
    allowed = {tool.name for tool in get_chat_tools(state["route"])}
    tool_messages = []
    results = list(state.get("tool_results", []))
    for call in state["planned_tool_calls"]:
        run = await execute_tool_call(
            call, state["conversation_id"], allowed_names=allowed,
        )
        tool_messages.append(run.tool_message)
        results.append({"tool_call_id": run.tool_call_id, "name": run.name,
                        "ok": run.ok, "content": str(run.tool_message.content)})
    return {"messages": tool_messages, "tool_results": results}


async def final_answer(state: dict, runtime: Runtime[dict]) -> dict:
    parts = []
    async for chunk in runtime.context["model"].astream(
        _fit_messages(state, GRAPH_FINAL_SYSTEM)
    ):
        if isinstance(chunk.content, str):
            parts.append(chunk.content)
    answer = "".join(parts).strip() or "暂时无法给出可靠答复，请稍后再试。"
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


async def chitchat_reply(state: dict) -> dict:
    answer = "您好，我是小喵。可以帮您了解商品、订单和售后问题。"
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


async def complaint_reply(state: dict) -> dict:
    answer = "很抱歉给您带来困扰。您可以选择转人工或填写工单，我们会按您的选择处理。"
    return {"answer": answer, "messages": [AIMessage(content=answer)],
            "suggested_actions": [
                {"type": "transfer_human"},
                {"type": "create_ticket", "draft": {"ticket_type": "投诉",
                                                    "description": state["query"]}},
            ]}


async def fallback_reply(state: dict) -> dict:
    answer = (FAQ_REFUSAL if state.get("route") == "knowledge" else
              "暂时无法确定您的需求，请换个说法或联系官方客服。")
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


async def log_turn(state: dict) -> dict:
    """MySQL remains the business audit log independent of SQLite checkpoints."""
    conversation_id = state["conversation_id"]
    marker = await repository.append_turn_messages(
        conversation_id, state["query"], state.get("tool_results", []), state["answer"],
    )
    return {"conversation_id": conversation_id,
            "trace": {"intent": state.get("intent"), "route": state.get("route"),
                      "steps": state.get("steps", 0), "tokens_used": state.get("tokens_used", 0),
                      "audit_message_id": marker}}


def _strong_evidence(payload: object) -> bool:
    if not isinstance(payload, dict) or payload.get("sufficient") is not True:
        return False
    citations = payload.get("citations")
    return (
        isinstance(payload.get("evidence"), str)
        and bool(payload["evidence"].strip())
        and isinstance(citations, list)
        and bool(citations)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("n"), int)
            and isinstance(item.get("answer"), str)
            and isinstance(item.get("section_path"), str)
            for item in citations
        )
    )


async def forced_rag(state: dict) -> dict:
    """Always retrieve first; malformed or failed evidence cannot enter Agent."""
    query = state["query"]
    try:
        payload = await query_faq.ainvoke({"keyword": query})
    except Exception as exc:  # noqa: BLE001 - retrieval failure is a refusal
        payload = {"sufficient": False, "source": "self_check",
                   "reason": f"知识检索失败: {type(exc).__name__}"}

    if _strong_evidence(payload):
        return {
            "sufficient": True, "evidence": payload["evidence"],
            "citations": payload["citations"], "reason": "",
        }

    source = payload.get("source") if isinstance(payload, dict) else None
    if source not in {"retrieval_low_conf", "self_check"}:
        source = "self_check"
    reason = payload.get("reason") if isinstance(payload, dict) else None
    if not isinstance(reason, str) or not reason.strip():
        reason = "知识库证据格式不完整或不足"
    await repository.insert_low_confidence(state["conversation_id"], query, source, reason)
    return {"sufficient": False, "evidence": "", "citations": [],
            "source": source, "reason": reason}


def confidence_gate(state: dict) -> str:
    return "agent" if _strong_evidence(state) else "fallback"
