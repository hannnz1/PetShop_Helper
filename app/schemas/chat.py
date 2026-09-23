from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(
        min_length=1,
        max_length=128,
        description="会话 ID,同一会话多轮复用",
    )
    message: str = Field(
        min_length=1,
        max_length=20_000,
        description="用户本轮消息",
    )
