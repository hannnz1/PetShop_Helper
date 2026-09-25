r"""Vectorize pending MySQL knowledge rows; safe to run again after interruption.

Windows: .\.venv\Scripts\python.exe -m scripts.vectorize_kb
Unix: uv run python -m scripts.vectorize_kb
"""

import asyncio

from app.kb import dualwrite, milvus_client


async def main() -> None:
    client = milvus_client.get_client()
    try:
        milvus_client.ensure_collection(client)
        count = await dualwrite.vectorize_pending(client)
        print(f"本次向量化 {count} 块；Milvus 现有 {milvus_client.count(client)} 条")
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
