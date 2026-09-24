"""Create the chat model for the configured OpenAI-compatible upstream."""

from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings


def get_chat_model(
    streaming: bool = False, settings: Settings | None = None
) -> ChatOpenAI:
    """Build a chat-completions client from the current application settings."""

    config = settings if settings is not None else get_settings()
    return ChatOpenAI(
        model=config.chat_model,
        base_url=config.chat_base_url,
        api_key=config.chat_api_key,
        timeout=config.chat_timeout,
        max_retries=0,
        max_tokens=config.chat_max_tokens,
        streaming=streaming,
        stream_usage=False,
        use_responses_api=False,
    )
