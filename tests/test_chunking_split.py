"""Markdown heading and Chinese-aware recursive splitting contracts."""

from app.kb import chunking


def test_split_sections_preserves_header_path():
    md = "# 售后手册\n\n## 退货政策\n\n支持7天无理由。\n\n## 运费说明\n\n满99包邮。"
    sections = chunking.split_sections(md)
    shipping = [section for section in sections if section.metadata.get("h2") == "运费说明"]
    assert len(shipping) == 1
    assert shipping[0].metadata["h1"] == "售后手册"
    assert "满99包邮" in shipping[0].page_content


def test_recursive_split_breaks_oversized_chinese_text():
    text = "句子内容。" * 60
    parts = chunking.recursive_split(text, chunk_size=50)
    assert len(parts) > 1
    assert all(0 < len(part) <= 50 for part in parts)
    assert "".join(parts) == text


def test_recursive_split_empty_text_returns_empty_list():
    assert chunking.recursive_split("", chunk_size=50) == []
