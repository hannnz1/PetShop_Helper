from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chat_model: str
    chat_base_url: str
    chat_api_key: SecretStr
    token_budget: int = Field(default=2000, gt=0)
    chat_max_tokens: int = Field(default=1024, gt=0)
    model_context_window: int = Field(default=32768, gt=0)
    max_output_tokens: int = Field(default=2000, gt=0)
    max_user_input_tokens: int = Field(default=2000, gt=0)
    tool_result_max_tokens: int = Field(default=1200, gt=0)
    expected_history_turns: int = Field(default=20, gt=0)
    steady_turn_chars: int = Field(default=280, gt=0)
    cjk_chars_per_token: float = Field(default=1.0, gt=0)
    layer2_assistant_chars: int = Field(default=48, gt=0)
    layer2_summary_tokens: int = Field(default=1695, gt=0)
    chat_timeout: float = Field(default=60, gt=0)
    max_agent_steps: int = Field(default=6, ge=1, le=12)
    graph_checkpoint_path: str = "data/ch05/checkpoints.sqlite"
    structured_output_method: Literal["json_mode", "json_schema"] = "json_mode"
    database_url: str = "mysql+asyncmy://root:root@127.0.0.1:3306/mewhelp?charset=utf8mb4"
    test_database_url: str = "mysql+asyncmy://root:root@127.0.0.1:3306/mewhelp_test?charset=utf8mb4"
    siliconflow_api_key: SecretStr | None = None
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_model: str = "BAAI/bge-m3"
    milvus_uri: str = "data/milvus_knowledge.db"
    retrieval_top_k: int = Field(default=3, gt=0)
    retrieval_min_score: float = Field(default=0.4, ge=0, le=1)
    rerank_api_key: SecretStr | None = None
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    recall_top_k: int = Field(default=50, gt=0)
    rerank_top_k: int = Field(default=10, gt=0)
    rerank_min_score: float = Field(default=0.3, ge=0, le=1)

    @field_validator("chat_model", "chat_base_url", "chat_api_key", mode="before")
    @classmethod
    def reject_blank_chat_settings(cls, value: object) -> object:
        raw_value = value.get_secret_value() if isinstance(value, SecretStr) else value
        if isinstance(raw_value, str) and not raw_value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("siliconflow_api_key", "rerank_api_key")
    @classmethod
    def reject_blank_embed_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("must not be blank")
        return value


@lru_cache
def get_settings() -> Settings:
    """Build settings on first use and reuse them for the process lifetime."""

    return Settings()
