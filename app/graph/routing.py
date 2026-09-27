"""Fixed exits and bounded ReAct decisions."""

from app.graph.state import ConversationState


INTENT_ROUTES = {
    "物流": "business", "订单": "business", "售后": "refund",
    "商品咨询": "knowledge", "退款退货": "refund",
    "投诉": "complaint", "闲聊": "chitchat", "人工": "business",
}


def route_by_intent(intent: str) -> str:
    return INTENT_ROUTES.get(intent, "fallback")


def should_continue(state: ConversationState, *, max_steps: int) -> str:
    if not state.get("planned_tool_calls"):
        return "final"
    if state.get("steps", 0) >= max_steps:
        return "fallback"
    return "tools"
