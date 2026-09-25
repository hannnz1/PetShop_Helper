"""Heading-aware and Chinese-aware text splitting primitives."""

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
