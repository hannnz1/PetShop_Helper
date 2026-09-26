"""Rerank request and response contract without spending API credit."""

from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from app.core import rerank as rr


def _settings(*, separate_key=None, account_key="account-key"):
    return SimpleNamespace(
        rerank_base_url="https://api.siliconflow.cn/v1",
        rerank_model="BAAI/bge-reranker-v2-m3",
        rerank_api_key=SecretStr(separate_key) if separate_key else None,
        siliconflow_api_key=SecretStr(account_key) if account_key else None,
    )


@pytest.mark.asyncio
async def test_rerank_orders_and_truncates_with_optional_separate_key(monkeypatch):
    monkeypatch.setattr(rr, "get_settings", lambda: _settings(separate_key="rerank-key"))
    calls = []

    async def fake_post(url, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        return httpx.Response(200, request=httpx.Request("POST", url), json={"results": [
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5},
        ]})

    monkeypatch.setattr(rr, "_post", fake_post)
    assert await rr.rerank("保修", ["甲", "乙", "丙"], top_n=2) == [(1, 0.9), (2, 0.5)]
    url, payload, headers, timeout = calls[0]
    assert url == "https://api.siliconflow.cn/v1/rerank"
    assert payload["model"] == "BAAI/bge-reranker-v2-m3"
    assert payload["query"] == "保修" and payload["documents"] == ["甲", "乙", "丙"]
    assert payload["top_n"] == 2
    assert headers["Authorization"] == "Bearer rerank-key"
    assert timeout > 0


@pytest.mark.asyncio
async def test_rerank_empty_docs_needs_no_key(monkeypatch):
    monkeypatch.setattr(rr, "get_settings", lambda: _settings(account_key=None))
    assert await rr.rerank("保修", []) == []


@pytest.mark.asyncio
async def test_rerank_falls_back_to_siliconflow_key(monkeypatch):
    monkeypatch.setattr(rr, "get_settings", _settings)

    async def fake_post(url, json, headers, timeout):
        assert headers["Authorization"] == "Bearer account-key"
        return httpx.Response(200, request=httpx.Request("POST", url), json={"results": [
            {"index": 0, "relevance_score": 0.6},
        ]})

    monkeypatch.setattr(rr, "_post", fake_post)
    assert await rr.rerank("保修", ["一年保修"]) == [(0, 0.6)]


@pytest.mark.asyncio
async def test_rerank_requires_key_only_when_called(monkeypatch):
    monkeypatch.setattr(rr, "get_settings", lambda: _settings(account_key=None))
    with pytest.raises(ValueError, match="SILICONFLOW_API_KEY or RERANK_API_KEY"):
        await rr.rerank("保修", ["一年保修"])


@pytest.mark.asyncio
async def test_rerank_rejects_untrusted_endpoint(monkeypatch):
    settings = _settings()
    settings.rerank_base_url = "http://example.test/v1"
    monkeypatch.setattr(rr, "get_settings", lambda: settings)
    with pytest.raises(ValueError, match="RERANK_BASE_URL"):
        await rr.rerank("保修", ["一年保修"])
