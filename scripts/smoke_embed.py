"""Live SiliconFlow BGE-M3 gate; prints only count and vector dimensions."""

import asyncio
import sys
from pathlib import Path
from urllib.parse import urlsplit

from openai import AsyncOpenAI

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings, get_settings


def _trusted_endpoint(base_url: str) -> bool:
    try:
        url = urlsplit(base_url)
        return (
            url.scheme == "https" and url.hostname == "api.siliconflow.cn"
            and url.port in (None, 443)
            and url.path.rstrip("/") == "/v1" and not url.username
            and not url.password and not url.query and not url.fragment
        )
    except ValueError:
        return False


async def run_smoke(settings: Settings, client_factory=AsyncOpenAI) -> bool:
    """Require the fixed provider/model and two valid 1024-D responses."""

    secret = settings.siliconflow_api_key
    key = secret.get_secret_value() if secret is not None else ""
    if not key or key.startswith("your-") or key.startswith("sk-your-"):
        print("NO-GO: 在本机 .env 配置 SILICONFLOW_API_KEY 后再运行真实嵌入冒烟。")
        return False
    if settings.embed_model != "BAAI/bge-m3" or not _trusted_endpoint(settings.embed_base_url):
        print("NO-GO: 第 3 章须使用 SiliconFlow HTTPS /v1 与 BAAI/bge-m3。")
        return False

    try:
        async with client_factory(
            api_key=key, base_url=settings.embed_base_url, timeout=30.0,
        ) as client:
            response = await client.embeddings.create(
                model=settings.embed_model, input=["邮费是多少", "运费怎么算"],
            )
        data = sorted(response.data, key=lambda item: item.index)
        dimensions = [len(item.embedding) for item in data]
        valid = len(data) == 2 and [item.index for item in data] == [0, 1] and dimensions == [1024, 1024]
        print(f"embedding_count={len(data)} dimensions={dimensions} result={'GO' if valid else 'NO-GO'}")
        return valid
    except Exception as exc:
        # Provider errors can include request URLs or tokens; report only type.
        print(f"NO-GO: 嵌入请求失败，异常类型 {type(exc).__name__}。")
        return False


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(run_smoke(get_settings())) else 1)
