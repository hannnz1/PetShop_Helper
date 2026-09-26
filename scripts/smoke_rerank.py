"""One small live rerank probe using the configured SiliconFlow account."""

import asyncio

from app.core.rerank import rerank


async def main() -> None:
    results = await rerank(
        "智能猫砂盆 Pro 的保修期是多久？",
        [
            "宠物零食满 99 元包邮。",
            "智能猫砂盆 Pro 提供一年保修。",
            "猫砂盆可选白色或灰色。",
        ],
        top_n=3,
    )
    assert results and results[0][0] == 1, f"rerank Top-1 mismatch: {[index for index, _ in results]}"
    print("GO: configured SiliconFlow reranker selected the warranty document at Top-1")


if __name__ == "__main__":
    asyncio.run(main())
