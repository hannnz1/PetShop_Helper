"""Structured single-label intent classification with a safe fallback."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from app.graph.routing import INTENT_ROUTES
from app.core.prompts import INTENT_PROMPT


class IntentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Literal["物流", "订单", "售后", "商品咨询", "退款退货", "投诉", "闲聊", "人工", "活动"]


class IntentClassifier(Protocol):
    async def classify(self, query: str) -> str: ...


class ModelIntentClassifier:
    def __init__(self, model, method: str = "json_mode") -> None:
        self.model = model
        self.method = method

    async def classify(self, query: str) -> str:
        chain = INTENT_PROMPT | self.model.with_structured_output(
            IntentResult, method=self.method,
        )
        parsed = IntentResult.model_validate(await chain.ainvoke({"query": query}))
        return parsed.intent

    async def classify_with_history(self, query: str, history: str) -> str:
        """Use the same bounded history view as coreference for short queries."""
        if not history:
            return await self.classify(query)
        payload = f"参考历史（只用于理解指代）：\n{history}\n当前用户消息：{query}"
        return await self.classify(payload)


async def safe_classify(classifier: IntentClassifier, query: str) -> str:
    """Malformed or unavailable classification never enters tool execution."""
    try:
        value = await classifier.classify(query)
    except Exception:  # noqa: BLE001 - fail closed at an upstream boundary
        return "unknown"
    return value if value in INTENT_ROUTES else "unknown"
