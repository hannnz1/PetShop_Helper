"""Whole knowledge workflows participate in the reset lifecycle lock."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from app.api import kb
from app.db import repository
from app.kb import mining
from scripts import build_kb


@pytest.mark.asyncio
async def test_build_mine_and_manual_ingest_hold_lifecycle_lock(monkeypatch, tmp_path):
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifecycle():
        async with lock:
            yield

    monkeypatch.setattr(repository, "knowledge_lifecycle_lock", lifecycle)

    source = tmp_path / "example.md"
    source.write_text("# 运费\n\n满 99 元包邮。", encoding="utf-8")
    monkeypatch.setattr(build_kb, "KB_DIR", tmp_path)
    monkeypatch.setattr(build_kb, "DOCS", {"example.md": "policy"})

    async def write_pending(_chunks):
        assert lock.locked()
        return [1]

    monkeypatch.setattr(build_kb.dualwrite, "write_pending", write_pending)
    await build_kb.main()

    async def mine_locked(_batch_size, _model):
        assert lock.locked()
        return {"sources": 0}

    monkeypatch.setattr(mining, "_mine_locked", mine_locked)
    assert await mining.mine() == {"sources": 0}

    async def ingest_locked(_request, _chunks):
        assert lock.locked()
        return {"count": 0}

    monkeypatch.setattr(kb, "_chunks", lambda _request: [])
    monkeypatch.setattr(kb, "_ingest_locked", ingest_locked)
    assert await kb.ingest(kb.SourceRequest(markdown="test", content_type="policy")) == {
        "count": 0
    }
