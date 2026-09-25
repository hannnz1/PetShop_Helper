# Ch03 reset API lookup (Context7, 2026-09-25)

- PyMilvus `/milvus-io/pymilvus`: synchronous `MilvusClient.has_collection(collection_name)` and `drop_collection(collection_name)` are the documented collection-management calls. Source: official PyMilvus API reference.
- SQLAlchemy `/websites/sqlalchemy_en_20`: `async with async_engine.begin() as conn: await conn.execute(...)` is the documented async transaction form. The official Core docs show deleting related tables in reverse dependency order.
- Reset implementation will only affect Chapter 3 `knowledge_chunks`, `qa_extraction_staging`, and Milvus `knowledge`. The CLI requires an explicit confirmation flag. No reset was run on the user's data.
