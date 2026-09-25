"""Async BGE-M3 embeddings through SiliconFlow's OpenAI-compatible endpoint."""

from openai import AsyncOpenAI
from urllib.parse import urlsplit

from app.config import get_settings


EMBED_DIMENSION = 1024


def _trusted_endpoint(base_url: str) -> bool:
    try:
        url = urlsplit(base_url)
        return (
            url.scheme == "https" and url.hostname == "api.siliconflow.cn"
            and url.port in (None, 443) and url.path.rstrip("/") == "/v1"
            and not url.username and not url.password and not url.query and not url.fragment
        )
    except ValueError:
        return False


def _client() -> AsyncOpenAI:
    settings = get_settings()
    secret = settings.siliconflow_api_key
    if secret is None or not secret.get_secret_value().strip():
        raise ValueError("SILICONFLOW_API_KEY is required for embeddings")
    if not _trusted_endpoint(settings.embed_base_url):
        raise ValueError("EMBED_BASE_URL must point to SiliconFlow HTTPS /v1")
    return AsyncOpenAI(
        base_url=settings.embed_base_url,
        api_key=secret.get_secret_value(),
        timeout=30.0,
    )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one 1024-D vector per input, in input order or raise."""
    if not texts:
        return []
    settings = get_settings()
    async with _client() as client:
        response = await client.embeddings.create(model=settings.embed_model, input=texts)
    data = sorted(response.data, key=lambda item: item.index)
    if len(data) != len(texts) or [item.index for item in data] != list(range(len(texts))):
        raise ValueError("embedding response count or indexes do not match inputs")
    vectors = [item.embedding for item in data]
    if any(len(vector) != EMBED_DIMENSION for vector in vectors):
        raise ValueError("embedding response dimension does not match BGE-M3")
    return vectors


async def embed_query(text: str) -> list[float]:
    """Embed a single search query."""
    return (await embed_texts([text]))[0]
