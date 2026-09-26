"""Chapter 5 workflow nodes, beginning with the mandatory knowledge gate."""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.config import get_settings
from app.core.agent import ContextBudgetExceeded, FAQ_REFUSAL
from app.core import coref as coref_service, query_understanding, retrieval, selfcheck
from app.core.intent import safe_classify
from app.core.memory import estimate_tokens
from app.core.prompts import GRAPH_AGENT_SYSTEM, GRAPH_FINAL_SYSTEM
from app.db import repository
from app.graph.routing import route_by_intent
from app.tools.business import query_faq
from app.tools.infra import execute_tool_call
from app.tools.registry import get_chat_tools


_ORDER_RE = re.compile(r"(?<!\d)(\d{4,})(?!\d)")


def _extract_order_id(query: str) -> str:
    match = _ORDER_RE.search(query)
    return match.group(1) if match else ""


async def fetch_order(state: dict) -> dict:
    """Read an owned sample order or pause for a safe user selection."""
    user_id = state["user_id"]
    order_id = state.get("order_id") or _extract_order_id(
        state.get("resolved_query") or state["query"],
    )
    order = await repository.get_owned_sample_order(user_id, str(order_id)) if order_id else None
    while order is None:
        orders = await repository.list_sample_orders(user_id)
        if not orders:
            return {"order_id": "", "order_data": {}, "no_orders": True,
                    "reason": "当前没有可供选择的演示订单"}
        # LangGraph resumes this node from its beginning; only reads precede it.
        order_id = interrupt({"type": "select_order", "orders": orders,
                              "conversation_id": state.get("conversation_id")})
        order = await repository.get_owned_sample_order(user_id, str(order_id))
    return {"order_id": str(order_id), "order_data": order, "no_orders": False}


def _history_text(messages: list, max_turns: int = 6, max_tokens: int | None = None) -> str:
    """Use recent complete turns, excluding this turn's final human message."""
    prior = messages[:-1] if messages and isinstance(messages[-1], HumanMessage) else messages
    pairs = []
    pending = None
    for message in prior:
        if isinstance(message, HumanMessage):
            pending = message
        elif isinstance(message, AIMessage) and not message.tool_calls and pending is not None:
            pairs.append((pending, message))
            pending = None
    budget = max_tokens if max_tokens is not None else max(0, get_settings().token_budget // 4)
    kept = []
    for user, assistant in reversed(pairs[-max_turns:]):
        candidate = [f"用户：{user.content}", f"客服：{assistant.content}", *kept]
        if estimate_tokens([HumanMessage(content="\n".join(candidate))]) > budget:
            break
        kept = candidate
    return "\n".join(kept)


async def coref(state: dict, runtime: Runtime[dict]) -> dict:
    """Resolve for downstream understanding while preserving the raw query."""
    model = runtime.context.get("model") if runtime.context else None
    if model is None:
        return {"resolved_query": state["query"]}
    history = _history_text(state.get("messages", []))
    resolved = await coref_service.resolve(state["query"], history, model)
    return {"resolved_query": resolved}


async def classify_intent_node(state: dict, runtime: Runtime[dict]) -> dict:
    classifier = runtime.context.get("classifier") if runtime.context else None
    intent = await safe_classify(classifier, state.get("resolved_query") or state["query"]) if classifier else "unknown"
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
    if state.get("route") == "refund" and state.get("order_data"):
        order = state["order_data"]
        system.append(SystemMessage(content=(
            f"本轮已核验本人演示订单：订单号 {order.get('order_id', '')}，"
            f"状态 {order.get('status', '')}，商品 {order.get('product', '')}。"
            "只能说可填写待人工审核申请，不能声称已退款或审核通过。"
        )))
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
    actions = list(state.get("suggested_actions", []))
    for call in state["planned_tool_calls"]:
        if isinstance(call, dict) and call.get("name") == "submit_refund":
            call_id = call.get("id") if isinstance(call.get("id"), str) else "invalid-refund-call"
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            proposed_id = str(args.get("order_id") or "")
            owned = await repository.get_owned_sample_order(state["user_id"], proposed_id)
            permitted = (
                state.get("route") == "refund" and _strong_evidence(state)
                and proposed_id == state.get("order_id") and owned is not None
            )
            if permitted:
                draft = {"order_id": proposed_id, "reason": args.get("reason")}
                if not any(item.get("type") == "refund_form" and item.get("draft", {}).get("order_id") == proposed_id
                           for item in actions):
                    actions.append({"type": "refund_form", "draft": draft})
                content = "已向用户展示退款申请表单，待用户确认；尚未提交申请。请停止调用其他工具。"
            else:
                content = "未核验本人订单或退款政策证据，本次不提供退款申请入口。"
                if (state.get("route") == "refund" and proposed_id != state.get("order_id")
                        and not any(item.get("type") == "select_order" for item in actions)):
                    actions.append({"type": "select_order",
                                    "mode": "new_turn",
                                    "orders": await repository.list_sample_orders(state["user_id"])})
            tool_messages.append(ToolMessage(
                content=content, tool_call_id=call_id, name="submit_refund",
                status="success" if permitted else "error",
            ))
            results.append({"tool_call_id": call_id, "name": "submit_refund",
                            "ok": permitted, "content": content})
            continue
        run = await execute_tool_call(
            call, state["conversation_id"], allowed_names=allowed,
        )
        tool_messages.append(run.tool_message)
        results.append({"tool_call_id": run.tool_call_id, "name": run.name,
                        "ok": run.ok, "content": str(run.tool_message.content)})
    return {"messages": tool_messages, "tool_results": results,
            "suggested_actions": actions}


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
    answer = ("当前没有可供选择的演示订单，请先在演示用户下导入订单。"
              if state.get("no_orders") else
              FAQ_REFUSAL if state.get("route") in {"knowledge", "refund"} else
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
    query = state.get("resolved_query") or state["query"]
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


async def retrieve_policy(state: dict, runtime: Runtime[dict]) -> dict:
    """Search owned-order policy evidence before any refund suggestion."""
    base = state.get("resolved_query") or state["query"]
    order_status = (state.get("order_data") or {}).get("status", "")
    seed = f"{base} {order_status}".strip()
    model = runtime.context.get("model") if runtime.context else None
    queries = await query_understanding.expand_queries(seed, model)
    merged: dict[object, dict] = {}
    try:
        for query in queries:
            for hit in await retrieval.search_knowledge(
                query, strategy="hybrid_rerank", bm25_query=query,
            ):
                if not isinstance(hit, dict) or "id" not in hit:
                    continue
                score = hit.get("rerank_score")
                if not isinstance(score, (int, float)):
                    continue
                prior = merged.get(hit["id"])
                if prior is None or score > prior["rerank_score"]:
                    merged[hit["id"]] = hit
        ranked = sorted(merged.values(), key=lambda item: item["rerank_score"], reverse=True)
        top_score = ranked[0]["rerank_score"] if ranked else 0.0
        if top_score < get_settings().rerank_min_score:
            source, reason = "retrieval_low_conf", f"政策证据不足(top={top_score:.3f})"
        else:
            selected = ranked[:3]
            evidence_texts = [f"{hit['question']} {hit['answer']}" for hit in selected]
            check = await selfcheck.check_sufficient(seed, evidence_texts)
            if not check["useful"]:
                source, reason = "self_check", check["reason"] or "政策证据不足"
            else:
                citations = [
                    {"n": index, "id": hit["id"], "section_path": hit["section_path"],
                     "question": hit["question"], "answer": hit["answer"],
                     "content_type": hit.get("content_type")}
                    for index, hit in enumerate(selected, 1)
                ]
                evidence = "\n".join(
                    f"[{item['n']}] {item['question']}: {item['answer']}" for item in citations
                )
                return {"sufficient": True, "evidence": evidence,
                        "citations": citations, "reason": ""}
    except Exception as exc:  # noqa: BLE001 - retrieval and self-check fail closed
        source, reason = "self_check", f"政策检索失败: {type(exc).__name__}"
    await repository.insert_low_confidence(state["conversation_id"], state["query"], source, reason)
    return {"sufficient": False, "evidence": "", "citations": [],
            "source": source, "reason": reason}
