"""Heading-aware and Chinese-aware text splitting primitives."""

import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
CJK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", "，", " ", ""]


def split_sections(md: str) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    return splitter.split_text(md)


def recursive_split(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CJK_SEPARATORS,
        is_separator_regex=False,
        length_function=len,
    )
    return splitter.split_text(text)


_SENTENCE_RE = re.compile(
    r'[^。！？!?…\n]*[。！？!?…]+[”’"」』）)\]]*|[^。！？!?…\n]*\n|[^。！？!?…\n]+$'
)


def _trailing_sentences(text: str, max_chars: int) -> str:
    sentences = [part for part in _SENTENCE_RE.findall(text) if part]
    chosen: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if chosen and total + len(sentence) > max_chars:
            break
        chosen.insert(0, sentence)
        total += len(sentence)
    return "".join(chosen)


def apply_sentence_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prefix later chunks with whole trailing sentences from the previous chunk."""
    if overlap < 0:
        raise ValueError("overlap must be nonnegative")
    if not chunks or overlap == 0:
        return list(chunks)
    out = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        out.append(_trailing_sentences(previous, overlap) + current)
    return out


_TABLE_SEP_RE = re.compile(r"^\s*\|\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*$")


def _table_cells(line: str) -> list[str] | None:
    row = line.strip()
    if not row.startswith("|"):
        return None
    row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return re.split(r"(?<!\\)\|", row)


def is_table_block(text: str) -> bool:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2 or not _TABLE_SEP_RE.fullmatch(lines[1]):
        return False
    header_cells = _table_cells(lines[0])
    if header_cells is None or len(header_cells) < 2:
        return False
    return all(
        (cells := _table_cells(line)) is not None and len(cells) == len(header_cells)
        for line in lines[1:]
    )


def split_table_rows(table_md: str, max_rows: int) -> list[str]:
    """Split table data rows into groups, repeating the header in each group."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if not is_table_block(table_md):
        raise ValueError("table_md must be a Markdown table")

    lines = [line for line in table_md.strip().splitlines() if line.strip()]
    header, separator, *rows = lines
    if len(rows) <= max_rows:
        return [table_md.strip()]
    return [
        "\n".join([header, separator, *rows[index : index + max_rows]])
        for index in range(0, len(rows), max_rows)
    ]
