"""Offline checks for the glm-5.2 go/no-go gate."""

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.config import Settings
from scripts.smoke_toolcall import run_smoke


def settings(model="glm-5.2", base_url="https://api.z.ai/api/paas/v4"):
    return Settings(
        chat_model=model,
        chat_base_url=base_url,
        chat_api_key=SecretStr("very-secret-test-key"),
    )


class FakeBoundModel:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.response


class FakeModel:
    def __init__(self, response=None, error=None):
        self.bound = FakeBoundModel(response, error)
        self.tools = None

    def bind_tools(self, tools):
        self.tools = tools
        return self.bound


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("gpt-4o-mini", "https://api.openai.com/v1"),
        ("glm-5.2", "https://api.openai.com/v1"),
        ("glm-5.2", "https://api.z.ai.evil.example/api/paas/v4"),
        ("glm-5.2", "https://api.z.ai/v1"),
    ],
)
async def test_rejects_wrong_configuration_before_client_or_network(
    model, base_url, capsys
):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        raise AssertionError("client must not be constructed")

    assert not await run_smoke(settings(model, base_url), model_factory=factory)
    assert calls == []
    captured = capsys.readouterr()
    assert "NO-GO" in captured.out
    assert "very-secret-test-key" not in captured.out + captured.err


async def test_no_tool_calls_is_no_go(capsys):
    model = FakeModel(SimpleNamespace(tool_calls=[]))
    assert not await run_smoke(settings(), model_factory=lambda **kwargs: model)
    assert "NO-GO" in capsys.readouterr().out
    assert model.tools[0].name == "add"
    assert len(model.bound.prompts) == 1


async def test_correct_structured_call_is_go(capsys):
    result = SimpleNamespace(
        tool_calls=[{"name": "add", "args": {"a": 23, "b": 19}, "id": "call-1"}]
    )
    model = FakeModel(result)
    assert await run_smoke(settings(), model_factory=lambda **kwargs: model)
    assert "GO" in capsys.readouterr().out


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(tool_calls=[{"name": "subtract", "args": {"a": 23, "b": 19}}]),
        SimpleNamespace(tool_calls=[{"name": "add", "args": {"a": 23}}]),
        SimpleNamespace(tool_calls=[{"name": "add", "args": "23,19"}]),
        SimpleNamespace(tool_calls=[{"name": "add", "args": {"a": True, "b": 19}}]),
        SimpleNamespace(tool_calls="not-a-list"),
        object(),
    ],
)
async def test_malformed_result_is_no_go(response, capsys):
    model = FakeModel(response)
    assert not await run_smoke(settings(), model_factory=lambda **kwargs: model)
    assert "NO-GO" in capsys.readouterr().out


async def test_upstream_exception_is_sanitized(capsys):
    model = FakeModel(error=RuntimeError("very-secret-test-key raw upstream body"))
    assert not await run_smoke(settings(), model_factory=lambda **kwargs: model)
    captured = capsys.readouterr()
    assert "NO-GO" in captured.out
    assert "very-secret-test-key" not in captured.out + captured.err
    assert "raw upstream body" not in captured.out + captured.err
