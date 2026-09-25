import pytest
from types import SimpleNamespace

from app.kb import milvus_client as mc


@pytest.fixture()
def client(tmp_path):
    c = mc.get_client(uri=str(tmp_path / "knowledge.db"))
    mc.ensure_collection(c)
    yield c
    c.close()


def _vec(seed: float) -> list[float]:
    vector = [0.0] * mc.DIM
    vector[0] = seed
    vector[1] = 1.0 - seed
    return vector


def test_upsert_then_search_returns_fields(client):
    mc.upsert_vectors(client, [
        {"id": 1, "vector": _vec(1.0), "question": "运费怎么算", "answer": "满99包邮"},
        {"id": 2, "vector": _vec(0.0), "question": "发货时效", "answer": "48小时内发货"},
    ])

    hits = mc.search(client, _vec(0.98), top_k=1)
    assert len(hits) == 1
    assert hits[0]["id"] == 1
    assert hits[0]["question"] == "运费怎么算"
    assert hits[0]["answer"] == "满99包邮"
    assert isinstance(hits[0]["score"], float)


def test_upsert_is_idempotent_by_pk(client):
    row = {"id": 1, "vector": _vec(1.0), "question": "q", "answer": "a"}
    mc.upsert_vectors(client, [row])
    mc.upsert_vectors(client, [row])
    assert mc.count(client) == 1


def test_ensure_collection_is_idempotent(client):
    mc.ensure_collection(client)
    assert mc.count(client) == 0


def test_count_loads_existing_released_collection():
    class ReleasedClient:
        loaded = False

        def load_collection(self, name):
            assert name == mc.COLLECTION
            self.loaded = True

        def query(self, name, *, filter, output_fields):
            assert (name, filter, output_fields) == (mc.COLLECTION, "id >= 0", ["count(*)"])
            if not self.loaded:
                raise RuntimeError("collection released")
            return [{"count(*)": 3}]

    assert mc.count(ReleasedClient()) == 3


def test_runtime_client_is_reused_and_closed_on_shutdown(monkeypatch):
    created = []

    class FakeClient:
        closed = False

        def close(self):
            self.closed = True

    def create(uri=None):
        created.append(uri)
        return FakeClient()

    monkeypatch.setattr(mc, "get_settings", lambda: SimpleNamespace(milvus_uri="runtime-test.db"))
    monkeypatch.setattr(mc, "get_client", create)
    first = mc.get_runtime_client()
    second = mc.get_runtime_client()
    assert first is second
    assert created == ["runtime-test.db"]
    mc.close_runtime_clients()
    assert first.closed


def test_runtime_client_stays_open_until_last_application_exits(monkeypatch):
    created = []

    class FakeClient:
        closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(mc, "get_settings", lambda: SimpleNamespace(milvus_uri="shared-app.db"))
    monkeypatch.setattr(mc, "get_client", lambda uri=None: created.append(FakeClient()) or created[-1])
    mc.add_runtime_owner()
    mc.add_runtime_owner()
    client = mc.get_runtime_client()
    mc.release_runtime_owner()
    assert not client.closed
    assert mc.get_runtime_client() is client
    mc.release_runtime_owner()
    assert client.closed


def test_empty_upsert_is_noop(client):
    mc.upsert_vectors(client, [])
    assert mc.count(client) == 0


def test_get_client_creates_parent_directory(tmp_path):
    path = tmp_path / "missing" / "knowledge.db"
    c = mc.get_client(uri=str(path))
    try:
        mc.ensure_collection(c)
        assert path.exists()
    finally:
        c.close()


def test_upsert_replaces_existing_row(client):
    mc.upsert_vectors(client, [{"id": 1, "vector": _vec(1.0), "question": "old", "answer": "old"}])
    mc.upsert_vectors(client, [{"id": 1, "vector": _vec(1.0), "question": "new", "answer": "new"}])
    assert mc.count(client) == 1
    assert mc.search(client, _vec(1.0), top_k=1)[0]["answer"] == "new"


def test_search_empty_collection_returns_empty_list(client):
    assert mc.search(client, _vec(1.0), top_k=2) == []


@pytest.mark.parametrize(
    ("dimension", "metric"),
    [(mc.DIM - 1, "COSINE"), (mc.DIM, "L2")],
)
def test_existing_incompatible_collection_is_rejected(tmp_path, dimension, metric):
    c = mc.get_client(uri=str(tmp_path / "incompatible.db"))
    try:
        c.create_collection(mc.COLLECTION, dimension=dimension, metric_type=metric)
        with pytest.raises(ValueError, match="incompatible"):
            mc.ensure_collection(c)
    finally:
        c.close()
