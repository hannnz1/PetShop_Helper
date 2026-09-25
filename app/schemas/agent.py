"""Request and response models for the programmatic agent endpoint."""

from pydantic import BaseModel, ConfigDict, Field


class AgentRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: int | None = Field(default=None, gt=0)


class ToolCallView(BaseModel):
    id: str
    name: str
    args: dict


class ToolResultView(BaseModel):
    tool_call_id: str
    name: str
    ok: bool
    content: str


class AgentResponse(BaseModel):
    conversation_id: int
    answer: str
    tool_calls: list[ToolCallView]
    tool_results: list[ToolResultView]
