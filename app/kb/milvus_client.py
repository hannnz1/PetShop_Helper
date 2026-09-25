"""Milvus Lite collection for knowledge-base vectors."""

from pathlib import Path

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from app.config import get_settings


COLLECTION = "knowledge"
DIM = 1024


def get_client(uri: str | None = None) -> MilvusClient:
    target = uri or get_settings().milvus_uri
    if target.endswith(".db"):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    return MilvusClient(uri=target)


def ensure_collection(client: MilvusClient) -> None:
    """Create the fixed schema once, with MySQL IDs as Milvus primary keys."""
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


def upsert_vectors(client: MilvusClient, rows: list[dict]) -> None:
    if not rows:
        return
    result = client.upsert(COLLECTION, data=rows)
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


def count(client: MilvusClient) -> int:
    return int(client.query(COLLECTION, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"])
