"""Direct SiliconFlow reranking for retrieved knowledge candidates."""

from math import isfinite
from urllib.parse import urlsplit

import httpx

from app.config import get_settings


def _trusted_base_url(base_url: str) -> bool:
    try:
        url = urlsplit(base_url)
        return (
            url.scheme == "https" and url.hostname == "api.siliconflow.cn"
            and url.port in (None, 443) and url.path.rstrip("/") == "/v1"
            and not url.username and not url.password and not url.query and not url.fragment
        )
    except ValueError:
        return False


async def _post(url: str, json: dict, headers: dict, timeout: float) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        return await client.post(url, json=json, headers=headers, timeout=timeout)


async def rerank(query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """Return original document indices and scores in descending relevance order."""
    if not docs:
        return []
    if not query.strip():
        raise ValueError("rerank query must not be blank")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be positive")

    settings = get_settings()
    if not _trusted_base_url(settings.rerank_base_url):
        raise ValueError("RERANK_BASE_URL must point to SiliconFlow HTTPS /v1")
    key = settings.rerank_api_key or settings.siliconflow_api_key
    if key is None or not key.get_secret_value().strip():
        raise ValueError("SILICONFLOW_API_KEY or RERANK_API_KEY is required for reranking")

    limit = min(top_n or len(docs), len(docs))
    response = await _post(
        settings.rerank_base_url.rstrip("/") + "/rerank",
        {
            "model": settings.rerank_model,
            "query": query,
            "documents": docs,
            "top_n": limit,
            "return_documents": False,
        },
        {"Authorization": f"Bearer {key.get_secret_value()}"},
        60.0,
    )
    response.raise_for_status()
    results = response.json()["results"]
    ranked = []
    seen = set()
    for item in results:
        index = item["index"]
        score = float(item["relevance_score"])
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(docs):
            raise ValueError("rerank response contains invalid document index")
        if index in seen or not isfinite(score):
            raise ValueError("rerank response contains duplicate index or invalid score")
        seen.add(index)
        ranked.append((index, score))
    return sorted(ranked, key=lambda result: result[1], reverse=True)[:limit]
