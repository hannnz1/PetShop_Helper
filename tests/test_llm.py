from pydantic import SecretStr

from app.config import Settings
from app.core.llm import get_chat_model


def test_factory_uses_injected_chat_settings() -> None:
    settings = Settings(
        _env_file=None,
        chat_model="test-chat-model",
        chat_base_url="https://chat.example.test/v1",
        chat_api_key=SecretStr("test-only-key"),
        chat_timeout=12.5,
        chat_max_tokens=321,
    )

    model = get_chat_model(settings=settings)

    assert model.model_name == settings.chat_model
    assert str(model.openai_api_base) == settings.chat_base_url
    assert model.openai_api_key.get_secret_value() == "test-only-key"
    assert model.request_timeout == settings.chat_timeout
    assert model.max_tokens == settings.chat_max_tokens
    assert model.max_retries == 0
    assert model.stream_usage is False
    assert model.use_responses_api is False
    assert model.temperature is None


def test_factory_passes_streaming_flag() -> None:
    settings = Settings(
        _env_file=None,
        chat_model="test-chat-model",
        chat_base_url="https://chat.example.test/v1",
        chat_api_key=SecretStr("test-only-key"),
    )

    assert get_chat_model(streaming=True, settings=settings).streaming is True
    assert get_chat_model(streaming=False, settings=settings).streaming is False


def test_stream_usage_requires_explicit_opt_in() -> None:
    settings = Settings(_env_file=None, chat_model="test-chat-model",
                        chat_base_url="https://chat.example.test/v1",
                        chat_api_key=SecretStr("test-only-key"), chat_stream_usage=True)
    assert get_chat_model(streaming=True, settings=settings).stream_usage is True


def test_factory_allows_deterministic_judge_without_changing_default() -> None:
    settings = Settings(
        _env_file=None,
        chat_model="test-chat-model",
        chat_base_url="https://chat.example.test/v1",
        chat_api_key=SecretStr("test-only-key"),
    )
    assert get_chat_model(settings=settings).temperature is None
    assert get_chat_model(settings=settings, temperature=0).temperature == 0
