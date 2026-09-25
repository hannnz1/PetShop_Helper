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
