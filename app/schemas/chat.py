"""Streaming chat request shape for persisted chapter-two conversations."""

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: int | None = Field(default=None, gt=0)
