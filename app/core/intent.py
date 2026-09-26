"""Safe intent-classifier boundary; the Prompt implementation arrives later."""

from typing import Protocol

from app.graph.routing import INTENT_ROUTES


class IntentClassifier(Protocol):
    async def classify(self, query: str) -> str: ...


async def safe_classify(classifier: IntentClassifier, query: str) -> str:
    """Malformed or unavailable classification never enters tool execution."""
    try:
        value = await classifier.classify(query)
    except Exception:  # noqa: BLE001 - fail closed at an upstream boundary
        return "unknown"
    return value if value in INTENT_ROUTES else "unknown"
