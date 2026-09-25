"""Mine reusable QA from chat history into pending MySQL knowledge chunks.

Run from the repository root: ``python -m scripts.mine_knowledge``.
Run ``python -m scripts.vectorize_kb`` afterwards to update Milvus Lite.
"""

import asyncio

from app.kb import mining


async def main() -> None:
    stats = await mining.mine()
    print(f"挖知识：{stats}")


if __name__ == "__main__":
    asyncio.run(main())
