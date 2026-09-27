from pydantic import SecretStr

from app.config import Settings


def test_langfuse_defaults_off_and_keys_hidden(monkeypatch):
    for name in (
        "LANGFUSE_ENABLED", "LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY", "OBSERVABILITY_ADMIN_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    disabled = Settings(
        chat_model="test-model", chat_base_url="https://example.test/v1",
        chat_api_key="test-key", _env_file=None,
    )
    assert disabled.langfuse_enabled is False
    assert disabled.langfuse_base_url == "http://127.0.0.1:3000"
    assert disabled.langfuse_public_key is None
    assert disabled.langfuse_secret_key is None
    assert disabled.observability_admin_token is None

    configured = Settings(
        chat_model="test-model", chat_base_url="https://example.test/v1",
        chat_api_key="test-key", langfuse_enabled=True,
        langfuse_public_key="lf_pk_example", langfuse_secret_key="lf_sk_example",
        observability_admin_token="admin-example", _env_file=None,
    )
    for secret in (configured.langfuse_public_key, configured.langfuse_secret_key,
                   configured.observability_admin_token):
        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value()
        assert secret.get_secret_value() not in repr(configured)
