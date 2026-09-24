"""Offline contracts for the after-sales extraction endpoint."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from app.api import extract as extract_api
from app.config import Settings
from app.core.memory import estimate_tokens
from app.core.prompts import EXTRACT_PROMPT
from app.main import create_app
from app.schemas.extract import AfterSalesTicket


def config(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        chat_model="test-model",
        chat_base_url="https://mock.invalid/v1",
        chat_api_key="test-only",
        **overrides,
    )


class StubModel:
    def __init__(self, runnable):
        self.runnable = runnable
        self.calls = []

    def with_structured_output(self, schema, *, method):
        self.calls.append((schema, method))
        return self.runnable


def test_extract_returns_exact_ticket_and_configured_method():
    inputs = []
    ticket = AfterSalesTicket(
        order_id="MH20260701123", request_type="退款", expected_solution="到货损坏要求退款"
    )
    model = StubModel(RunnableLambda(lambda value: inputs.append(value) or ticket))
    app = create_app(settings=config(structured_output_method="json_schema"), model=model)
    with TestClient(app) as client:
        response = client.post("/api/extract", json={"text": "订单 MH20260701123 损坏了，请退款"})
    assert response.status_code == 200
    assert response.json() == {
        "order_id": "MH20260701123",
        "request_type": "退款",
        "expected_solution": "到货损坏要求退款",
    }
    assert model.calls == [(AfterSalesTicket, "json_schema")]
    assert len(inputs) == 1
    assert inputs[0].messages[1].content == "订单 MH20260701123 损坏了，请退款"


def test_extract_omitted_order_id_is_null():
    model = StubModel(RunnableLambda(lambda _: {
        "order_id": None, "request_type": "其他", "expected_solution": "未明确"
    }))
    app = create_app(settings=config(), model=model)
    with TestClient(app) as client:
        response = client.post("/api/extract", json={"text": "东西有问题"})
    assert response.status_code == 200
    assert response.json() == {
        "order_id": None, "request_type": "其他", "expected_solution": "未明确"
    }


@pytest.mark.parametrize("bad_output", [None, {}, {"request_type": "无效", "expected_solution": "退款"}])
def test_extract_malformed_upstream_output_is_sanitized_502(bad_output, caplog):
    model = StubModel(RunnableLambda(lambda _: bad_output))
    app = create_app(settings=config(), model=model)
    with TestClient(app) as client:
        response = client.post("/api/extract", json={"text": "请退款"})
    assert response.status_code == 502
    assert response.json() == {"detail": "上游模型暂时不可用，请稍后重试"}
    assert "bad-output-secret" not in caplog.text


def test_extract_upstream_failure_is_sanitized_502(caplog):
    def boom(_):
        raise RuntimeError("private-key upstream response body")

    app = create_app(settings=config(), model=StubModel(RunnableLambda(boom)))
    with TestClient(app) as client:
        response = client.post("/api/extract", json={"text": "请退款"})
    assert response.status_code == 502
    assert response.json() == {"detail": "上游模型暂时不可用，请稍后重试"}
    assert "private-key" not in response.text + caplog.text


def test_extract_binding_failure_is_sanitized_502(caplog):
    class BrokenBinding:
        def with_structured_output(self, schema, *, method):
            raise RuntimeError("private-key unsupported structured output method")

    app = create_app(settings=config(), model=BrokenBinding())
    with TestClient(app) as client:
        response = client.post("/api/extract", json={"text": "请退款"})
    assert response.status_code == 502
    assert response.json() == {"detail": "上游模型暂时不可用，请稍后重试"}
    assert "private-key" not in response.text + caplog.text


def test_extract_empty_and_over_budget_rejected_before_model_call():
    calls = []
    model = StubModel(RunnableLambda(lambda _: calls.append(1)))
    app = create_app(settings=config(token_budget=500), model=model)
    with TestClient(app) as client:
        assert client.post("/api/extract", json={"text": " "}).status_code == 422
        response = client.post("/api/extract", json={"text": "长" * 600})
    assert response.status_code == 422
    assert calls == []


def test_extract_budget_counts_rendered_system_and_user_text():
    text = "我要退款"
    full_prompt_tokens = estimate_tokens(EXTRACT_PROMPT.format_messages(text=text))
    calls = []
    model = StubModel(RunnableLambda(lambda _: calls.append(1)))
    app = create_app(settings=config(token_budget=full_prompt_tokens - 1), model=model)
    with TestClient(app) as client:
        response = client.post("/api/extract", json={"text": text})
    assert response.status_code == 422
    assert calls == []


def test_extract_dependency_can_be_overridden_offline():
    app = create_app(settings=config(), model=object())
    ticket = AfterSalesTicket(order_id=None, request_type="投诉", expected_solution="希望处理投诉")
    app.dependency_overrides[extract_api.get_extractor] = lambda: RunnableLambda(lambda _: ticket)
    try:
        with TestClient(app) as client:
            response = client.post("/api/extract", json={"text": "我要投诉"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {
        "order_id": None, "request_type": "投诉", "expected_solution": "希望处理投诉"
    }


@pytest.mark.asyncio
async def test_real_chatopenai_json_mode_uses_response_format_without_tools():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={
            "id": "chatcmpl-offline",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps({
                    "order_id": None,
                    "request_type": "换货",
                    "expected_solution": "更换损坏商品",
                }, ensure_ascii=False)},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        model = ChatOpenAI(
            model="test-model", api_key="test-only", base_url="https://mock.invalid/v1",
            http_async_client=upstream, max_retries=0, use_responses_api=False,
        )
        app = create_app(settings=config(), model=model)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/api/extract", json={"text": "东西坏了，想换货"})

    assert response.status_code == 200
    assert response.json() == {
        "order_id": None, "request_type": "换货", "expected_solution": "更换损坏商品"
    }
    assert len(seen) == 1
    path, body = seen[0]
    assert path.endswith("/chat/completions")
    assert body["response_format"] == {"type": "json_object"}
    assert "tools" not in body
    assert body["messages"][0]["role"] == "system"
    assert "order_id" in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": "东西坏了，想换货"}
