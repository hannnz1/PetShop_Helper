"""Fail-fast smoke test for native BM25 and dense/RRF on Milvus Standalone."""

from __future__ import annotations

import uuid

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker


def main() -> None:
    client = MilvusClient(uri="http://127.0.0.1:19530")
    collection = f"ch04_smoke_{uuid.uuid4().hex[:12]}"
    try:
        analysis = client.run_analyzer(
            texts=["MH-W40", "LP100", "Pro"],
            analyzer_params={"type": "chinese"},
        )
        print(f"Chinese analyzer output: {analysis}")
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
                {"id": 4, "text": "LP100 宠物饮水机提供一年保修。", "dense": [0.0, 0.0, 0.0, 1.0]},
                {"id": 5, "text": "Pro 款自动喂食器支持定时投喂。", "dense": [0.5, 0.5, 0.0, 0.0]},
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
        for query, expected in [("LP100 保修", 4), ("Pro 定时投喂", 5)]:
            hits = client.search(
                collection_name=collection,
                data=[query],
                anns_field="sparse",
                limit=3,
                output_fields=["text"],
            )[0]
            assert hits and hits[0]["id"] == expected, f"BM25 {query} mismatch: {hits}"

        dense = client.search(
            collection_name=collection,
            data=[[1.0, 0.0, 0.0, 0.0]],
            anns_field="dense",
            limit=3,
            output_fields=["text"],
        )[0]
        assert dense and dense[0]["id"] == 1, f"Dense mismatch: {dense}"

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
        filtered = client.search(
            collection_name=collection,
            data=["MH-W40 保质期"],
            anns_field="sparse",
            filter="id != 1",
            limit=3,
            output_fields=["text"],
        )[0]
        assert filtered and all(hit["id"] != 1 for hit in filtered), f"Filter mismatch: {filtered}"

        client.upsert(
            collection_name=collection,
            data={"id": 1, "text": "MH-W40 狗粮保质期十八个月，未拆封可退。", "dense": [1.0, 0.0, 0.0, 0.0]},
        )
        client.flush(collection_name=collection)
        updated = client.get(collection_name=collection, ids=[1], output_fields=["text"])
        assert len(updated) == 1 and "未拆封可退" in updated[0]["text"], f"Upsert mismatch: {updated}"
        print("GO: Milvus Standalone native Chinese BM25, dense/RRF, filter, and upsert")
    finally:
        if client.has_collection(collection):
            client.drop_collection(collection)


if __name__ == "__main__":
    main()
