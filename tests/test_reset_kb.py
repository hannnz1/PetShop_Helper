"""Reset CLI boundaries; never operate on the configured real stores here."""

import pytest

from scripts import reset_kb


def test_reset_cli_requires_explicit_confirmation():
    with pytest.raises(SystemExit) as error:
        reset_kb.main([])
    assert error.value.code == 2


@pytest.mark.asyncio
async def test_reset_only_drops_knowledge_collection_and_closes_client(monkeypatch):
    events = []

    class FakeClient:
        def has_collection(self, name):
            events.append(("has", name))
            return True

        def drop_collection(self, name):
            events.append(("drop", name))

        def close(self):
            events.append(("close",))

    async def fake_reset(drop_vectors):
        events.append(("mysql-lock",))
        drop_vectors()

    monkeypatch.setattr(reset_kb.milvus_client, "get_client", FakeClient)
    monkeypatch.setattr(reset_kb.repository, "reset_knowledge_tables", fake_reset)
    await reset_kb.reset()
    assert events == [
        ("mysql-lock",),
        ("has", reset_kb.milvus_client.COLLECTION),
        ("drop", reset_kb.milvus_client.COLLECTION),
        ("close",),
    ]
