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
    chat_timeout: float = Field(default=60, gt=0)
    structured_output_method: Literal["json_mode", "json_schema"] = "json_mode"
    database_url: str = "mysql+asyncmy://root:root@127.0.0.1:3306/mewhelp?charset=utf8mb4"
    test_database_url: str = "mysql+asyncmy://root:root@127.0.0.1:3306/mewhelp_test?charset=utf8mb4"

    @field_validator("chat_model", "chat_base_url", "chat_api_key", mode="before")
    @classmethod
    def reject_blank_chat_settings(cls, value: object) -> object:
        raw_value = value.get_secret_value() if isinstance(value, SecretStr) else value
        if isinstance(raw_value, str) and not raw_value.strip():
            raise ValueError("must not be blank")
        return value


@lru_cache
def get_settings() -> Settings:
    """Build settings on first use and reuse them for the process lifetime."""

    return Settings()
