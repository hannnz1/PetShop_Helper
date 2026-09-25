r"""Ingest the three reviewed Markdown fixtures into pending MySQL knowledge.

Windows: .\.venv\Scripts\python.exe -m scripts.build_kb
Unix: uv run python -m scripts.build_kb
"""

import asyncio
from dataclasses import replace
from pathlib import Path

from app.kb import documents, dualwrite, sources


KB_DIR = sources.KB_DIR
DOCS = sources.SOURCE_TYPES


def source_chunks(filename: str, markdown: str, content_type: str) -> list[documents.Chunk]:
    """Carry source identity in the existing section_path field for idempotence."""
    path = Path(filename)
    if path.name != filename or filename in ("", ".", "..") or not filename.endswith(".md"):
        raise ValueError("source must be a Markdown filename within data/kb")
    source = f"data/kb/{filename}"
    return [
        replace(chunk, section_path=f"{source} :: {chunk.section_path}")
        for chunk in documents.build_chunks(markdown, content_type=content_type)
    ]


async def main() -> None:
    total = 0
    for filename, kind in DOCS.items():
        markdown = (KB_DIR / filename).read_text(encoding="utf-8")
        chunks = source_chunks(filename, markdown, kind)
        ids = await dualwrite.write_pending(chunks)
        total += len(ids)
        print(f"{filename}: {len(ids)} 块")
    print(f"本次建库处理 {total} 块；向量化请运行 make kb-vectorize")


if __name__ == "__main__":
    asyncio.run(main())
