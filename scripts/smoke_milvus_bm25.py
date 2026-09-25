"""Fail-fast smoke test for native BM25 and dense/RRF on Milvus Standalone."""

from __future__ import annotations

import uuid

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker


def main() -> None:
    client = MilvusClient(uri="http://127.0.0.1:19530")
    collection = f"ch04_smoke_{uuid.uuid4().hex[:12]}"
    try:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=2048,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
        schema.add_function(
            Function(
                name="bm25_text",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )
        indexes = client.prepare_index_params()
        indexes.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        indexes.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(collection_name=collection, schema=schema, index_params=indexes)
        client.insert(
            collection_name=collection,
            data=[
                {"id": 1, "text": "MH-W40 狗粮的保质期为十八个月。", "dense": [1.0, 0.0, 0.0, 0.0]},
                {"id": 2, "text": "MH-P20 猫砂七天内支持退货。", "dense": [0.0, 1.0, 0.0, 0.0]},
                {"id": 3, "text": "宠物项圈有蓝色和红色。", "dense": [0.0, 0.0, 1.0, 0.0]},
            ],
        )
        client.flush(collection_name=collection)
        client.load_collection(collection_name=collection)

        bm25 = client.search(
            collection_name=collection,
            data=["MH-W40 保质期"],
            anns_field="sparse",
            limit=3,
            output_fields=["text"],
        )[0]
        assert bm25 and bm25[0]["id"] == 1, f"BM25 Chinese/model-term mismatch: {bm25}"

        requests = [
            AnnSearchRequest(data=["MH-W40 保质期"], anns_field="sparse", param={"metric_type": "BM25"}, limit=3),
            AnnSearchRequest(data=[[1.0, 0.0, 0.0, 0.0]], anns_field="dense", param={"metric_type": "COSINE"}, limit=3),
        ]
        hybrid = client.hybrid_search(
            collection_name=collection,
            reqs=requests,
            ranker=RRFRanker(k=60),
            limit=3,
            output_fields=["text"],
        )[0]
        assert hybrid and hybrid[0]["id"] == 1, f"Hybrid RRF mismatch: {hybrid}"
        print("GO: Milvus Standalone native Chinese BM25 and dense/RRF return MH-W40 at Top-1")
    finally:
        if client.has_collection(collection):
            client.drop_collection(collection)


if __name__ == "__main__":
    main()
