"""Preview local Markdown knowledge chunks without model or database calls."""

from scripts.build_kb import DOCS, KB_DIR, source_chunks


def main() -> None:
    total = 0
    for filename, kind in DOCS.items():
        markdown = (KB_DIR / filename).read_text(encoding="utf-8")
        chunks = source_chunks(filename, markdown, kind)
        total += len(chunks)
        print(f"{filename}: {len(chunks)} 块，关键条款 {sum(c.is_key_clause for c in chunks)} 块")
        for chunk in chunks:
            print(f"  {chunk.questions}: {len(chunk.answer)} 字")
    print(f"合计 {total} 块")


if __name__ == "__main__":
    main()
