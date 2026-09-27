"""Run one bounded canonicalization batch (no raw-question output)."""

import argparse
import asyncio

from app.core.llm import get_chat_model
from app.flywheel.canonicalize import canonicalize_batch


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--allow-external-real-text", action="store_true")
    args = parser.parse_args()
    result = await canonicalize_batch(
        args.limit, get_chat_model(temperature=0),
        allow_external_real_text=args.allow_external_real_text,
    )
    print(f"processed={result.processed} merged={result.merged} new={result.new} "
          f"pending_upstream={result.pending_upstream}")


if __name__ == "__main__":
    asyncio.run(main())
