"""Knowledge vectors: legacy Lite access plus isolated Chapter 4 hybrid collections."""

import json
from pathlib import Path
from threading import Lock

from pymilvus import (
    AnnSearchRequest, CollectionSchema, DataType, FieldSchema, Function,
    FunctionType, MilvusClient, RRFRanker,
)

from app.config import get_settings


COLLECTION = "knowledge"
DIM = 1024
HYBRID_OUTPUT = ["question", "answer", "section_path", "content_type", "category"]
_runtime_lock = Lock()
_runtime_clients: dict[str, MilvusClient] = {}
_runtime_owners = 0


def get_client(uri: str | None = None) -> MilvusClient:
    target = uri or get_settings().milvus_uri
    if target.endswith(".db"):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    return MilvusClient(uri=target)


def get_runtime_client() -> MilvusClient:
    """Keep one local connection per URI during the web process lifetime."""
    uri = get_settings().milvus_uri
    with _runtime_lock:
        client = _runtime_clients.get(uri)
        if client is None:
            client = get_client(uri=uri)
            _runtime_clients[uri] = client
        return client


def close_runtime_clients() -> None:
    """Force-release cached clients, primarily for standalone use and tests."""
    with _runtime_lock:
        _close_runtime_clients_locked()


def _close_runtime_clients_locked() -> None:
    for client in _runtime_clients.values():
        client.close()
    _runtime_clients.clear()


def add_runtime_owner() -> None:
    """Register one active FastAPI application in this process."""
    global _runtime_owners
    with _runtime_lock:
        _runtime_owners += 1


def release_runtime_owner() -> None:
    """Close shared clients only after the final application exits."""
    global _runtime_owners
    with _runtime_lock:
        if _runtime_owners < 1:
            raise RuntimeError("No runtime Milvus owner to release")
        _runtime_owners -= 1
        if _runtime_owners == 0:
            _close_runtime_clients_locked()


def ensure_collection(client: MilvusClient, collection: str = COLLECTION) -> None:
    """Create the fixed schema once, with MySQL IDs as Milvus primary keys."""
    if collection != COLLECTION:
        _ensure_hybrid_collection(client, collection)
        return
    if client.has_collection(COLLECTION):
        _validate_collection(client)
        client.load_collection(COLLECTION)
        return

    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema("question", DataType.VARCHAR, max_length=2048),
            FieldSchema("answer", DataType.VARCHAR, max_length=8192),
        ],
        enable_dynamic_field=False,
    )
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
    )
    client.create_collection(COLLECTION, schema=schema, index_params=index_params)
    client.load_collection(COLLECTION)


def _ensure_hybrid_collection(client: MilvusClient, collection: str) -> None:
    """Build native BM25/dense schema in an isolated collection before migration."""
    if client.has_collection(collection):
        _validate_hybrid_collection(client, collection)
        client.load_collection(collection)
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field(
        "text", DataType.VARCHAR, max_length=8192,
        enable_analyzer=True, analyzer_params={"type": "chinese"},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    for name, length in (
        ("question", 2048), ("answer", 8192), ("section_path", 512),
        ("content_type", 32), ("category", 255),
    ):
        schema.add_field(name, DataType.VARCHAR, max_length=length)
    schema.add_function(Function(
        name="text_bm25", function_type=FunctionType.BM25,
        input_field_names=["text"], output_field_names=["sparse"],
    ))
    indexes = client.prepare_index_params()
    indexes.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
    indexes.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(collection_name=collection, schema=schema, index_params=indexes)
    client.load_collection(collection)


def _validate_hybrid_collection(client: MilvusClient, collection: str) -> None:
    description = client.describe_collection(collection)
    fields = {field["name"]: field for field in description["fields"]}
    expected = {
        "id": DataType.INT64, "dense": DataType.FLOAT_VECTOR,
        "text": DataType.VARCHAR, "sparse": DataType.SPARSE_FLOAT_VECTOR,
        **{name: DataType.VARCHAR for name in HYBRID_OUTPUT},
    }
    if description.get("auto_id") or any(
        name not in fields or fields[name]["type"] != dtype
        for name, dtype in expected.items()
    ):
        raise ValueError("Existing Milvus hybrid collection has incompatible schema")
    if not fields["id"].get("is_primary") or int(fields["dense"]["params"].get("dim", 0)) != DIM:
        raise ValueError("Existing Milvus hybrid collection has incompatible vector fields")
    functions = description.get("functions", [])
    if not any(
        fn.get("type") == FunctionType.BM25
        and fn.get("input_field_names") == ["text"]
        and fn.get("output_field_names") == ["sparse"]
        for fn in functions
    ):
        raise ValueError("Existing Milvus hybrid collection has incompatible BM25 function")
    for field_name, metric in (("dense", "COSINE"), ("sparse", "BM25")):
        names = client.list_indexes(collection, field_name=field_name)
        if not names or client.describe_index(collection, names[0]).get("metric_type") != metric:
            raise ValueError("Existing Milvus hybrid collection has incompatible indexes")


def _validate_collection(client: MilvusClient) -> None:
    """Reject stale collections rather than silently searching incompatible vectors."""
    description = client.describe_collection(COLLECTION)
    fields = {field["name"]: field for field in description["fields"]}
    expected_types = {
        "id": DataType.INT64,
        "vector": DataType.FLOAT_VECTOR,
        "question": DataType.VARCHAR,
        "answer": DataType.VARCHAR,
    }
    if description.get("auto_id") or any(
        name not in fields or fields[name]["type"] != dtype
        for name, dtype in expected_types.items()
    ):
        raise ValueError("Existing Milvus knowledge collection has incompatible schema")
    if (
        not fields["id"].get("is_primary")
        or int(fields["vector"]["params"].get("dim", 0)) != DIM
        or int(fields["question"]["params"].get("max_length", 0)) < 2048
        or int(fields["answer"]["params"].get("max_length", 0)) < 8192
    ):
        raise ValueError("Existing Milvus knowledge collection has incompatible fields")
    indexes = client.list_indexes(COLLECTION, field_name="vector")
    if not indexes:
        raise ValueError("Existing Milvus knowledge collection has incompatible index")
    index = client.describe_index(COLLECTION, indexes[0])
    if (
        index is None
        or index.get("field_name") != "vector"
        or index.get("index_type") != "AUTOINDEX"
        or index.get("metric_type") != "COSINE"
    ):
        raise ValueError("Existing Milvus knowledge collection has incompatible index")


def upsert_vectors(client: MilvusClient, rows: list[dict], collection: str = COLLECTION) -> None:
    if not rows:
        return
    result = client.upsert(collection, data=rows)
    if result.get("upsert_count") != len(rows):
        raise RuntimeError("Milvus upsert count differs from submitted row count")


def search(client: MilvusClient, vector: list[float], top_k: int) -> list[dict]:
    results = client.search(
        COLLECTION,
        data=[vector],
        limit=top_k,
        output_fields=["question", "answer"],
        search_params={"metric_type": "COSINE"},
    )
    return [
        {
            "id": hit["id"],
            "score": float(hit["distance"]),
            "question": hit["entity"]["question"],
            "answer": hit["entity"]["answer"],
        }
        for hit in results[0]
    ]


def count(client: MilvusClient, collection: str = COLLECTION) -> int:
    client.load_collection(collection)
    return int(client.query(collection, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"])


def flush(client: MilvusClient, collection: str = COLLECTION) -> None:
    client.flush(collection)


def drop(client: MilvusClient, collection: str) -> None:
    if client.has_collection(collection):
        client.drop_collection(collection)


def _category_filter(category: str | None) -> str:
    return f"category == {json.dumps(category, ensure_ascii=False)}" if category else ""


def _hybrid_hits(results: list[list[dict]]) -> list[dict]:
    return [
        {
            "id": hit["id"], "score": float(hit["distance"]),
            **{field: hit["entity"][field] for field in HYBRID_OUTPUT},
        }
        for hit in results[0]
    ]


def dense_search(
    client: MilvusClient, vector: list[float], top_k: int,
    category: str | None = None, collection: str = COLLECTION,
) -> list[dict]:
    results = client.search(
        collection_name=collection, data=[vector], anns_field="dense", limit=top_k,
        output_fields=HYBRID_OUTPUT, search_params={"metric_type": "COSINE"},
        filter=_category_filter(category),
    )
    # Standalone COSINE score is similarity: an identical vector is near 1.
    return _hybrid_hits(results)


def bm25_search(
    client: MilvusClient, text: str, top_k: int,
    category: str | None = None, collection: str = COLLECTION,
) -> list[dict]:
    results = client.search(
        collection_name=collection, data=[text], anns_field="sparse", limit=top_k,
        output_fields=HYBRID_OUTPUT, search_params={"metric_type": "BM25"},
        filter=_category_filter(category),
    )
    return _hybrid_hits(results)


def hybrid_search(
    client: MilvusClient, vector: list[float], text: str, top_k: int,
    recall: int = 50, category: str | None = None, collection: str = COLLECTION,
) -> list[dict]:
    expr = _category_filter(category)
    requests = [
        AnnSearchRequest(
            data=[vector], anns_field="dense", param={"metric_type": "COSINE"},
            limit=recall, filter=expr,
        ),
        AnnSearchRequest(
            data=[text], anns_field="sparse", param={"metric_type": "BM25"},
            limit=recall, filter=expr,
        ),
    ]
    results = client.hybrid_search(
        collection_name=collection, reqs=requests, ranker=RRFRanker(k=60),
        limit=top_k, output_fields=HYBRID_OUTPUT,
    )
    return _hybrid_hits(results)
