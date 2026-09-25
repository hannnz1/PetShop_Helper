import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings, get_settings


REQUIRED_CHAT_ENV = ("CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY")


@pytest.mark.parametrize("missing_name", REQUIRED_CHAT_ENV)
def test_each_chat_account_setting_is_required(monkeypatch, missing_name):
    values = {
        "CHAT_MODEL": "test-model",
        "CHAT_BASE_URL": "https://example.test/v1",
        "CHAT_API_KEY": "test-key",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(missing_name)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    missing_fields = {
        error["loc"][0]
        for error in exc_info.value.errors()
        if error["type"] == "missing"
    }
    assert missing_fields == {missing_name.lower()}


@pytest.mark.parametrize("field_name", ("chat_model", "chat_base_url", "chat_api_key"))
@pytest.mark.parametrize("blank_value", ("", " \t "))
def test_chat_account_settings_reject_blank_values(field_name, blank_value):
    values = {
        "chat_model": "test-model",
        "chat_base_url": "https://example.test/v1",
        "chat_api_key": "test-key",
        field_name: blank_value,
    }

    with pytest.raises(ValidationError) as exc_info:
        Settings(**values, _env_file=None)

    assert exc_info.value.errors()[0]["loc"] == (field_name,)


def test_defaults_and_secret_type():
    settings = Settings(
        chat_model="test-model",
        chat_base_url="https://example.test/v1",
        chat_api_key="test-key",
        _env_file=None,
    )

    assert settings.token_budget == 2000
    assert settings.chat_max_tokens == 1024
    assert settings.chat_timeout == 60
    assert settings.structured_output_method == "json_mode"
    assert isinstance(settings.chat_api_key, SecretStr)
    assert settings.chat_api_key.get_secret_value() == "test-key"
    assert "test-key" not in repr(settings)


def test_environment_overrides_defaults(monkeypatch):
    values = {
        "CHAT_MODEL": "env-model",
        "CHAT_BASE_URL": "https://env.example/v1",
        "CHAT_API_KEY": "env-key",
        "TOKEN_BUDGET": "500",
        "CHAT_MAX_TOKENS": "256",
        "CHAT_TIMEOUT": "15.5",
        "STRUCTURED_OUTPUT_METHOD": "json_schema",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)

    assert settings.token_budget == 500
    assert settings.chat_max_tokens == 256
    assert settings.chat_timeout == 15.5
    assert settings.structured_output_method == "json_schema"


@pytest.mark.parametrize("field_name", ("token_budget", "chat_max_tokens", "chat_timeout"))
@pytest.mark.parametrize("invalid_value", (0, -1))
def test_limits_must_be_positive(field_name, invalid_value):
    values = {
        "chat_model": "test-model",
        "chat_base_url": "https://example.test/v1",
        "chat_api_key": "test-key",
        field_name: invalid_value,
    }

    with pytest.raises(ValidationError) as exc_info:
        Settings(**values, _env_file=None)

    assert exc_info.value.errors()[0]["loc"] == (field_name,)


def test_structured_output_method_rejects_unsupported_value():
    with pytest.raises(ValidationError):
        Settings(
            chat_model="test-model",
            chat_base_url="https://example.test/v1",
            chat_api_key="test-key",
            structured_output_method="function_calling",
            _env_file=None,
        )


def test_get_settings_is_cached(monkeypatch):
    values = {
        "CHAT_MODEL": "cached-model",
        "CHAT_BASE_URL": "https://cached.example/v1",
        "CHAT_API_KEY": "cached-key",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second
    get_settings.cache_clear()


def test_ch03_embedding_settings_are_optional_until_live_gate(monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    settings = Settings(
        chat_model="test-model", chat_base_url="https://example.test/v1",
        chat_api_key="test-key", _env_file=None,
    )
    assert settings.siliconflow_api_key is None
    assert settings.embed_base_url == "https://api.siliconflow.cn/v1"
    assert settings.embed_model == "BAAI/bge-m3"
    assert settings.milvus_uri == "data/milvus_knowledge.db"
    assert settings.retrieval_top_k == 3
    assert settings.retrieval_min_score == 0.4


def test_ch03_embedding_key_uses_design_documents_name(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-embed-secret")
    settings = Settings(
        chat_model="test-model", chat_base_url="https://example.test/v1",
        chat_api_key="test-key", _env_file=None,
    )
    assert isinstance(settings.siliconflow_api_key, SecretStr)
    assert settings.siliconflow_api_key.get_secret_value() == "test-embed-secret"
    assert "test-embed-secret" not in repr(settings)


def test_ch04_retrieval_defaults_preserve_pre_migration_milvus(monkeypatch):
    monkeypatch.delenv("RERANK_API_KEY", raising=False)
    settings = Settings(
        chat_model="test-model", chat_base_url="https://example.test/v1",
        chat_api_key="test-key", _env_file=None,
    )
    assert settings.milvus_uri == "data/milvus_knowledge.db"
    assert settings.rerank_api_key is None
    assert settings.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert settings.recall_top_k == 50
    assert settings.rerank_top_k == 10
    assert settings.rerank_min_score == 0.3


def test_ch04_retrieval_settings_load_from_environment(monkeypatch):
    overrides = {
        "MILVUS_URI": "http://127.0.0.1:19530",
        "RERANK_API_KEY": "test-rerank-key",
        "RERANK_MODEL": "BAAI/test-reranker",
        "RECALL_TOP_K": "30",
        "RERANK_TOP_K": "7",
        "RERANK_MIN_SCORE": "0.45",
    }
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    settings = Settings(
        chat_model="test-model", chat_base_url="https://example.test/v1",
        chat_api_key="test-key", _env_file=None,
    )
    assert settings.milvus_uri == overrides["MILVUS_URI"]
    assert settings.rerank_api_key.get_secret_value() == "test-rerank-key"
    assert "test-rerank-key" not in repr(settings)
    assert settings.rerank_model == "BAAI/test-reranker"
    assert settings.recall_top_k == 30
    assert settings.rerank_top_k == 7
    assert settings.rerank_min_score == 0.45


@pytest.mark.parametrize("field_name", ("recall_top_k", "rerank_top_k"))
def test_ch04_top_k_must_be_positive(field_name):
    with pytest.raises(ValidationError):
        Settings(
            chat_model="test-model", chat_base_url="https://example.test/v1",
            chat_api_key="test-key", **{field_name: 0}, _env_file=None,
        )


@pytest.mark.parametrize("value", (-0.1, 1.1))
def test_ch04_min_score_must_be_probability(value):
    with pytest.raises(ValidationError):
        Settings(
            chat_model="test-model", chat_base_url="https://example.test/v1",
            chat_api_key="test-key", rerank_min_score=value, _env_file=None,
        )
