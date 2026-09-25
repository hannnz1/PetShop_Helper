"""Vectorization cannot overlap knowledge reset, including a failed embed."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.kb import dualwrite


@pytest.mark.asyncio
async def test_vectorize_holds_knowledge_lock_through_embedding(monkeypatch):
    lock = asyncio.Lock()
    embedding_started = asyncio.Event()
    release_embedding = asyncio.Event()

    @asynccontextmanager
    async def knowledge_lock():
        async with lock:
            yield

    async def pending():
        return [SimpleNamespace(id=1, category="物流", questions="运费", answer="包邮")]

    async def embedding(_texts):
        embedding_started.set()
        await release_embedding.wait()
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr(dualwrite.repository, "knowledge_write_lock", knowledge_lock)
    monkeypatch.setattr(dualwrite.repository, "list_pending_chunks", pending)
    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", embedding)
    vectorizing = asyncio.create_task(dualwrite.vectorize_pending(None))
    try:
        await asyncio.wait_for(embedding_started.wait(), 1)
        reset_lock = asyncio.create_task(lock.acquire())
        await asyncio.sleep(0)
        assert not reset_lock.done()
        release_embedding.set()
        with pytest.raises(RuntimeError, match="upstream unavailable"):
            await vectorizing
        await asyncio.wait_for(reset_lock, 1)
        lock.release()
    finally:
        release_embedding.set()
        if not vectorizing.done():
            await vectorizing
