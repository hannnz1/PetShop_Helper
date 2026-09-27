"""A Graph turn keeps one safe, locally scoped trace across each entry point."""

import logging

import pytest
from langgraph.graph import START, StateGraph
from pydantic import SecretStr
from typing import TypedDict

from app.config import Settings


def _settings(**overrides):
    values = {"chat_model": "test-model", "chat_base_url": "https://example.test/v1",
              "chat_api_key": "test-key"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_disabled_missing_keys_and_cloud_never_construct_callback(monkeypatch):
    from app.observability import tracing

    def forbidden(*args, **kwargs):
        raise AssertionError("remote client must not be initialized")

    monkeypatch.setattr(tracing, "_new_handler", forbidden)
    assert tracing.make_turn_callbacks(_settings(), "alice", 7, "turn-1") == []
    assert tracing.make_turn_callbacks(_settings(langfuse_enabled=True), "alice", 7, "turn-1") == []
    cloud = _settings(langfuse_enabled=True, langfuse_base_url="https://cloud.langfuse.com",
                      langfuse_public_key="pk", langfuse_secret_key="sk")
    assert tracing.make_turn_callbacks(cloud, "alice", 7, "turn-1") == []


@pytest.mark.parametrize("field", ["langfuse_public_key", "langfuse_secret_key"])
def test_blank_extracted_key_never_constructs_client(monkeypatch, field):
    from app.observability import tracing

    constructed = []
    monkeypatch.setattr(tracing, "_new_handler", lambda **kwargs: constructed.append(kwargs) or object())
    valid = _settings(langfuse_enabled=True, langfuse_public_key="pk",
                      langfuse_secret_key="sk")
    unchecked = valid.model_copy(update={field: SecretStr("  ")})
    assert tracing.make_turn_callbacks(unchecked, "alice", 7, "turn-1") == []
    assert constructed == []


def test_local_callback_uses_pseudonym_and_never_puts_credentials_in_metadata(monkeypatch):
    from app.observability import tracing

    created = []
    monkeypatch.setattr(tracing, "_new_handler", lambda **kwargs: created.append(kwargs) or object())
    settings = _settings(langfuse_enabled=True, langfuse_public_key="pk-secret",
                         langfuse_secret_key="sk-secret")
    callbacks = tracing.make_turn_callbacks(settings, "alice@example.test", 7, "turn-1")
    assert len(callbacks) == 1
    assert created[0]["public_key"] == "pk-secret"
    assert created[0]["secret_key"] == "sk-secret"
    metadata = tracing.turn_metadata(settings, "alice@example.test", 7, "turn-1")
    assert metadata["langfuse_session_id"] == "7"
    assert metadata["turn_id"] == "turn-1"
    assert metadata["langfuse_user_id"] != "alice@example.test"
    assert metadata["langfuse_user_id"] == tracing.turn_metadata(
        settings, "alice@example.test", 8, "turn-2")["langfuse_user_id"]
    assert "pk-secret" not in str(metadata) and "sk-secret" not in str(metadata)


def test_trace_mask_redacts_configured_secrets_and_authorization_headers():
    from app.observability import tracing

    settings = _settings(chat_api_key="model-secret", langfuse_enabled=True,
                         langfuse_public_key="pk-secret", langfuse_secret_key="sk-secret",
                         database_url="mysql+asyncmy://root:db-secret@127.0.0.1/app")
    mask = tracing.make_trace_mask(settings)
    payload = {"input": ["use model-secret", "mysql+asyncmy://root:db-secret@127.0.0.1/app"],
               "headers": {"Authorization": "Bearer private", "x-api-key": "other"}}
    result = mask(payload)
    assert result["input"] == ["use [REDACTED]", "[REDACTED]"]
    assert result["headers"] == {"Authorization": "[REDACTED]", "x-api-key": "[REDACTED]"}


@pytest.mark.asyncio
async def test_invoke_stream_and_resume_attach_one_trace(monkeypatch, caplog):
    monkeypatch.setenv("CHAT_MODEL", "test-model")
    monkeypatch.setenv("CHAT_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("CHAT_API_KEY", "test-key")
    from app.graph.runtime import GraphRuntime
    from app.observability import tracing

    class RecordingGraph:
        def __init__(self):
            self.configs = []

        async def ainvoke(self, value, config, **kwargs):
            self.configs.append(config)
            return {"trace": {}}

        async def astream(self, value, config, **kwargs):
            self.configs.append(config)
            yield ("updates", {"done": True})

    async def conversation(user_id, conversation_id):
        return 7

    async def snapshot(*args):
        return object()

    async def audit(*args, **kwargs):
        return None

    async def last_message(*args):
        return None

    monkeypatch.setattr("app.graph.runtime.repository.last_message_id", last_message)
    callback_turn_ids = []
    def callbacks(settings, user_id, conversation_id, turn_id):
        callback_turn_ids.append(turn_id)
        return [object()]
    monkeypatch.setattr("app.graph.runtime.make_turn_callbacks", callbacks)
    runtime = GraphRuntime("unused.sqlite", lambda saver: None, settings=_settings())
    runtime.graph = RecordingGraph()
    runtime._conversation_id = conversation
    runtime._prepare_context_snapshot = snapshot
    runtime._check_audit = audit
    with caplog.at_level(logging.INFO, logger="app.graph.runtime"):
        await runtime.ainvoke_turn("alice", "one", 7, model=object())
        stream = await runtime.prepare_stream_turn("alice", "two", 7, model=object())
        assert [event async for event in stream] == [("updates", {"done": True})]
        resume = await runtime.prepare_resume_turn("alice", 7, "1001", model=object())
        assert [event async for event in resume] == [("updates", {"done": True})]
    configs = runtime.graph.configs
    assert len(configs) == 3
    ids = [config["metadata"]["turn_id"] for config in configs]
    assert len(set(ids)) == 3
    assert callback_turn_ids == ids
    turn_logs = [record.getMessage() for record in caplog.records
                 if record.name == "app.graph.runtime" and "Graph turn started" in record.getMessage()]
    assert turn_logs == [f"Graph turn started turn_id={turn_id} conversation_id=7"
                         for turn_id in ids]
    assert all("alice" not in message and "one" not in message for message in turn_logs)
    assert all(config["configurable"]["thread_id"] == "7" for config in configs)
    assert all(len(config["callbacks"]) == 1 for config in configs)
    assert all(config["metadata"]["langfuse_session_id"] == "7" for config in configs)


@pytest.mark.asyncio
async def test_callback_failure_does_not_change_graph_result_or_stream(monkeypatch, caplog):
    from app.observability import tracing

    class BrokenHandler:
        def on_chain_start(self, *args, **kwargs):
            raise RuntimeError("telemetry down")

    monkeypatch.setattr(tracing, "_new_handler", lambda **kwargs: BrokenHandler())
    settings = _settings(langfuse_enabled=True, langfuse_public_key="pk",
                         langfuse_secret_key="sk")
    handler = tracing.make_turn_callbacks(settings, "alice", 7, "turn-1")[0]
    assert handler.on_chain_start({}, {}, run_id="run") is None
    class State(TypedDict):
        answer: str

    async def answer(state: State) -> State:
        return {"answer": "ok"}

    builder = StateGraph(State)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    graph = builder.compile()
    config = {"callbacks": [handler], "metadata": tracing.turn_metadata(settings, "alice", 7, "turn-1")}
    assert (await graph.ainvoke({"answer": ""}, config))["answer"] == "ok"
    assert [item async for item in graph.astream({"answer": ""}, config)]
    assert "Langfuse callback on_chain_start failed (RuntimeError)" in caplog.text
    assert "telemetry down" not in caplog.text


def test_shutdown_failure_is_logged_without_raising(monkeypatch, caplog):
    from app.observability import tracing

    class BrokenClient:
        def shutdown(self):
            raise RuntimeError("secret in exception")

    monkeypatch.setattr(tracing, "_get_client", lambda **kwargs: BrokenClient())
    tracing._initialized_keys.add("pk")
    settings = _settings(langfuse_enabled=True, langfuse_public_key="pk",
                         langfuse_secret_key="sk")
    tracing.close_tracing(settings)
    assert "Langfuse shutdown failed (RuntimeError)" in caplog.text
    assert "secret in exception" not in caplog.text
