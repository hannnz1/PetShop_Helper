"""Explicitly clear Chapter 3 MySQL knowledge and Milvus collection only.

Run from the project root: ``python -m scripts.reset_kb --yes``.
No conversations, messages, tickets, or Chapter 2 FAQ rows are touched.
"""

import argparse
import asyncio

from app.db import repository
from app.kb import milvus_client


async def reset() -> None:
    client = milvus_client.get_client()
    try:
        def drop_vectors() -> None:
            if client.has_collection(milvus_client.COLLECTION):
                client.drop_collection(milvus_client.COLLECTION)

        await repository.reset_knowledge_tables(drop_vectors)
    finally:
        client.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="confirm clearing Chapter 3 data")
    args = parser.parse_args(argv)
    if not args.yes:
        parser.error("reset requires --yes")
    asyncio.run(reset())
    print("Chapter 3 knowledge and staging cleared; Milvus collection dropped")


if __name__ == "__main__":
    main()
