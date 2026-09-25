"""Offline contract tests for the live evaluation command-line tools."""

import json
import sys

import httpx
import pytest

from scripts import demo_chat, eval_extract, eval_prompts


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")


@pytest.mark.parametrize("module", [eval_extract, demo_chat, eval_prompts])
def test_live_cli_clients_ignore_proxy_environment(monkeypatch, module):
    """Every live CLI must connect directly to loopback despite inherited proxies."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setattr(sys, "argv", ["live-cli"])
    kwargs_seen = []

    class FakeClient:
        def __init__(self, **kwargs):
            kwargs_seen.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(eval_extract, "require_live_config", lambda: None)
    monkeypatch.setattr(eval_extract, "SAMPLES", type("P", (), {"read_text": lambda *a, **k: "[]"})())
    monkeypatch.setattr(eval_prompts, "require_live_config", lambda: None)
    monkeypatch.setattr(eval_prompts, "CASES", type("P", (), {"read_text": lambda *a, **k: '{"cases": []}'})())
    monkeypatch.setattr(demo_chat, "run_demo", lambda *args: None)
    monkeypatch.setattr(eval_extract.httpx, "Client", FakeClient)

    assert module.main() == 0
    assert len(kwargs_seen) == 1
    assert kwargs_seen[0]["base_url"] == "http://127.0.0.1:8000"
    assert kwargs_seen[0]["trust_env"] is False


def test_extract_evaluator_reports_pass_and_failure(capsys):
    samples = [
        {"text": "退款", "expected": {"order_id": None, "request_type": "退款"}},
        {"text": "换货", "expected": {"order_id": "MH1", "request_type": "换货"}},
    ]

    def handler(request):
        text = json.loads(request.content)["text"]
        return httpx.Response(200, json={"order_id": None, "request_type": "退款", "expected_solution": text})

    with _client(handler) as client:
        assert eval_extract.evaluate(samples, client) == 1
    output = capsys.readouterr().out
    assert "[PASS] #1" in output
    assert "[FAIL] #2" in output
    assert "1/2 通过" in output


@pytest.mark.parametrize("response", [
    httpx.Response(200, text="not-json"),
    httpx.Response(200, json={"order_id": None, "request_type": "退款"}),
    httpx.Response(503, text="upstream unavailable"),
])
def test_extract_evaluator_fails_on_malformed_or_http_error(response, capsys):
    samples = [{"text": "退款", "expected": {"order_id": None, "request_type": "退款"}}]
    with _client(lambda request: response) as client:
        assert eval_extract.evaluate(samples, client) == 1
    assert "[FAIL] #1" in capsys.readouterr().out


def test_extract_evaluator_fails_on_transport_error(capsys):
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with _client(handler) as client:
        assert eval_extract.evaluate([{"text": "退款", "expected": {"order_id": None, "request_type": "退款"}}], client) == 1
    assert "连接" in capsys.readouterr().out


@pytest.mark.parametrize("stream", [
    'data: {bad json}\n\n',
    'event: error\ndata: {"message":"failed"}\n\n',
    'data: {"delta":"你好"}\n\n',
])
def test_demo_rejects_malformed_error_and_truncated_stream(stream):
    with _client(lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, text=stream)) as client:
        with pytest.raises(demo_chat.StreamError):
            demo_chat.stream_turn(client, "uid", None, "你好", lambda text: None)


def test_demo_accepts_complete_stream_and_two_turns(capsys):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text='data: {"delta":"你好"}\n\ndata: {"event":"done","conversation_id":1}\n\ndata: [DONE]\n\n')

    with _client(handler) as client:
        demo_chat.run_demo(client, "sid")
    assert len(sent) == 2
    assert sent[0]["user_id"] == sent[1]["user_id"] == "sid"
    assert sent[0]["conversation_id"] is None
    assert sent[1]["conversation_id"] == 1
    assert "你好" in capsys.readouterr().out


@pytest.mark.parametrize("response", [
    httpx.Response(502, text="bad gateway"),
    httpx.Response(200, headers={"content-type": "application/json"}, json={"delta": "你好"}),
])
def test_demo_rejects_http_and_content_type_errors(response):
    with _client(lambda request: response) as client:
        with pytest.raises(demo_chat.StreamError):
            demo_chat.stream_turn(client, "uid", None, "你好", lambda text: None)


def test_prompt_evaluator_prints_human_review_without_auto_quality_claim(capsys):
    cases = [{"id": "case1", "endpoint": "extract", "prompt": "退款", "expected_criteria": ["不得编造订单号"]}]
    with _client(lambda request: httpx.Response(200, json={"order_id": None, "request_type": "退款", "expected_solution": "退款"})) as client:
        assert eval_prompts.evaluate(cases, client) == 0
    output = capsys.readouterr().out
    assert "不得编造订单号" in output
    assert "人工复核" in output
    assert "通过" not in output


def test_live_config_refuses_missing_and_placeholder(tmp_path):
    path = tmp_path / ".env"
    with pytest.raises(eval_extract.ConfigurationError):
        eval_extract.require_live_config(path)
    path.write_text("CHAT_MODEL=YOUR_MODEL_NAME\nCHAT_BASE_URL=https://api.openai.com/v1\nCHAT_API_KEY=sk-your-api-key\n", encoding="utf-8")
    with pytest.raises(eval_extract.ConfigurationError):
        eval_extract.require_live_config(path)


@pytest.mark.asyncio
async def test_offline_socket_first_frame_arrives_before_producer_release():
    from scripts.check_chat_socket import check_socket_stream

    await check_socket_stream()
