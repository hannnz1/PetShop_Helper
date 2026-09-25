"""Turn source Markdown into reviewable knowledge chunks."""

from dataclasses import dataclass
import re

from app.kb import chunking


_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_KEY_MARKER = "<!-- key-clause -->"


@dataclass
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: int = 0


def _raw_sections(md: str):
    """Retain blank lines lost by the header splitter so markers scope one block."""
    path: dict[int, str] = {}
    body: list[str] = []
    fence_char: str | None = None
    fence_length = 0
    for line in md.splitlines():
        fence = _FENCE.match(line)
        if fence_char is not None:
            body.append(line)
            if fence and fence.group(1)[0] == fence_char and len(fence.group(1)) >= fence_length:
                fence_char = None
            continue
        if fence:
            fence_char = fence.group(1)[0]
            fence_length = len(fence.group(1))
            body.append(line)
            continue
        heading = _HEADING.match(line)
        if heading:
            if any(part.strip() for part in body):
                yield path.copy(), "\n".join(body)
            level = len(heading.group(1))
            path = {depth: title for depth, title in path.items() if depth < level}
            path[level] = heading.group(2).strip()
            body = []
        else:
            body.append(line)
    if any(part.strip() for part in body):
        yield path, "\n".join(body)


def build_chunks(
    md: str, content_type: str,
    chunk_size: int = 400, overlap: int = 60, table_max_rows: int = 10,
) -> list[Chunk]:
    """Apply an explicit marker only to its following paragraph or table block."""
    out: list[Chunk] = []
    for raw_path, body in _raw_sections(md):
        # Let the official Markdown splitter interpret the heading path; retain
        # the raw body separately because it discards paragraph blank lines.
        prefix = "\n".join(f"{'#' * level} {title}" for level, title in raw_path.items())
        split = chunking.split_sections(f"{prefix}\n\n{body}" if prefix else body)
        metadata = split[-1].metadata if split else {}
        path = [metadata[key] for key in ("h1", "h2", "h3", "h4") if metadata.get(key)]
        title = path[-1] if path else content_type
        category = " / ".join(path[:-1]) if len(path) > 1 else (path[0] if path else content_type)
        section_path = " / ".join(path)
        is_key = False
        for block in re.split(r"\n\s*\n", body.strip()):
            block = block.strip()
            if not block:
                continue
            if block == _KEY_MARKER:
                is_key = True
                continue
            if _KEY_MARKER in block:
                raise ValueError("key-clause marker must be a standalone block")
            if chunking.is_table_block(block):
                pieces = chunking.split_table_rows(block, table_max_rows)
            else:
                pieces = chunking.apply_sentence_overlap(
                    chunking.recursive_split(block, chunk_size, chunk_overlap=0), overlap
                )
            for piece in pieces:
                out.append(Chunk(
                    category=category, questions=title, answer=piece,
                    section_path=section_path, content_type=content_type,
                    is_key_clause=int(is_key),
                ))
            is_key = False
        if is_key:
            raise ValueError("key-clause marker has no following paragraph or table")
    return out
