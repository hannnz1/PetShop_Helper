"""Chapter 4 hybrid search against an isolated real Standalone collection."""

import uuid

import pytest

from app.kb import milvus_client as mc


def _vector(axis: int) -> list[float]:
    values = [0.0] * mc.DIM
    values[axis] = 1.0
    return values


@pytest.fixture(scope="module")
def hybrid_collection():
    client = mc.get_client(uri="http://127.0.0.1:19530")
    name = f"test_ch04_{uuid.uuid4().hex[:12]}"
    try:
        mc.ensure_collection(client, collection=name)
        mc.upsert_vectors(
            client,
            [
                {
                    "id": 1, "dense": _vector(0), "text": "运费满99元包邮，否则10元",
                    "question": "运费怎么算", "answer": "满99元包邮，否则10元",
                    "section_path": "运费政策", "content_type": "faq", "category": "运费",
                },
                {
                    "id": 2, "dense": _vector(1), "text": "智能猫砂盆 Pro 型号支持自动清理",
                    "question": "Pro 型号功能", "answer": "自动清理",
                    "section_path": "商品手册 / 猫砂盆", "content_type": "manual", "category": "商品手册",
                },
            ],
            collection=name,
        )
        mc.flush(client, collection=name)
        assert mc.count(client, collection=name) == 2
        yield client, name
    finally:
        mc.drop(client, name)
        client.close()


def test_bm25_hits_model_number(hybrid_collection):
    client, name = hybrid_collection
    mc.ensure_collection(client, collection=name)
    hits = mc.bm25_search(client, "猫砂盆 Pro 型号", top_k=2, collection=name)
    assert hits and hits[0]["id"] == 2
    assert hits[0]["section_path"] == "商品手册 / 猫砂盆"


def test_dense_and_hybrid_return_scored_hits(hybrid_collection):
    client, name = hybrid_collection
    dense = mc.dense_search(client, _vector(0), top_k=2, collection=name)
    assert dense and dense[0]["id"] == 1
    assert dense[0]["score"] > 0.99
    hits = mc.hybrid_search(client, _vector(0), "运费", top_k=2, recall=10, collection=name)
    assert hits and hits[0]["id"] == 1
    assert {"id", "score", "question", "answer", "section_path", "content_type", "category"} <= hits[0].keys()


def test_category_filter_is_applied_to_both_routes(hybrid_collection):
    client, name = hybrid_collection
    for search in (
        lambda: mc.bm25_search(client, "运费 猫砂盆", top_k=5, category="运费", collection=name),
        lambda: mc.dense_search(client, _vector(1), top_k=5, category="运费", collection=name),
        lambda: mc.hybrid_search(client, _vector(1), "运费 猫砂盆", top_k=5, category="运费", collection=name),
    ):
        hits = search()
        assert hits and {hit["id"] for hit in hits} == {1}


def test_existing_incompatible_hybrid_collection_is_rejected():
    client = mc.get_client(uri="http://127.0.0.1:19530")
    name = f"test_ch04_bad_{uuid.uuid4().hex[:12]}"
    try:
        client.create_collection(collection_name=name, dimension=mc.DIM)
        with pytest.raises(ValueError, match="incompatible"):
            mc.ensure_collection(client, collection=name)
    finally:
        mc.drop(client, name)
        client.close()
