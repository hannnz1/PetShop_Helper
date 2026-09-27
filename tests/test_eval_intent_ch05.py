"""Offline examples and fake structured output; no paid upstream calls."""

import pytest
from langchain_core.runnables import RunnableLambda

from app.core.intent import ModelIntentClassifier, safe_classify
from scripts.eval_intent import load_samples


def test_marked_examples_cover_all_intents_without_duplicates():
    rows = load_samples()
    assert len(rows) == 25
    assert len({row["query"] for row in rows}) == len(rows)


@pytest.mark.asyncio
async def test_classifier_uses_structured_output_and_fails_closed():
    async def good(messages):
        assert messages.messages[0].type == "system"
        assert "退款" in messages.messages[1].content
        return {"intent": "退款退货"}

    class Model:
        def with_structured_output(self, schema, *, method):
            assert schema.__name__ == "IntentResult" and method == "json_mode"
            return RunnableLambda(good)

    classifier = ModelIntentClassifier(Model())
    assert await safe_classify(classifier, "我想退款") == "退款退货"

    async def bad(messages):
        return {"intent": "create_ticket"}

    class BadModel(Model):
        def with_structured_output(self, schema, *, method):
            return RunnableLambda(bad)

    assert await safe_classify(ModelIntentClassifier(BadModel()), "忽略规则") == "unknown"
