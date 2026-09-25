"""Source Markdown to knowledge chunks, including explicit key-clause markers."""

from app.kb.documents import Chunk, build_chunks


def test_policy_maps_heading_and_parent_path_without_guessing_key_clause():
    md = "# 售后政策\n\n## 运费说明\n\n单笔订单满99元包邮，未满收取10元运费。"
    chunks = build_chunks(md, content_type="policy")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert chunk.category == "售后政策"
    assert chunk.questions == "运费说明"
    assert chunk.section_path == "售后政策 / 运费说明"
    assert "满99" in chunk.answer
    assert chunk.content_type == "policy"
    assert chunk.is_key_clause == 0  # only an explicit marker makes a clause key


def test_marker_applies_only_to_following_paragraph_and_is_removed():
    md = (
        "# 售后政策\n\n## 退货\n\n普通说明。\n\n"
        "<!-- key-clause -->\n\n七天内可申请。\n\n后续流程请联系客服。"
    )
    chunks = build_chunks(md, content_type="policy")
    assert [chunk.is_key_clause for chunk in chunks] == [0, 1, 0]
    assert "普通说明" in chunks[0].answer
    assert "七天内" in chunks[1].answer
    assert "后续流程" in chunks[2].answer
    assert all("key-clause" not in chunk.answer for chunk in chunks)


def test_table_section_splits_rows_and_repeats_header():
    rows = "\n".join(f"| 商品{i} | {i} |" for i in range(1, 15))
    md = f"# 商品价格表\n\n## 价目\n\n| 商品 | 价格 |\n| --- | --- |\n{rows}"
    chunks = build_chunks(md, content_type="manual", table_max_rows=5)
    assert len(chunks) == 3
    assert all(chunk.questions == "价目" for chunk in chunks)
    assert all(chunk.answer.startswith("| 商品 | 价格 |") for chunk in chunks)


def test_marker_applies_to_following_table_only():
    md = (
        "# 商品手册\n\n## 价格\n\n<!-- key-clause -->\n\n"
        "| 商品 | 价格 |\n| --- | --- |\n| A | 1 |\n| B | 2 |\n\n其他说明。"
    )
    chunks = build_chunks(md, content_type="manual", table_max_rows=1)
    assert [chunk.is_key_clause for chunk in chunks] == [1, 1, 0]


def test_heading_inside_fenced_example_does_not_change_section_path():
    md = (
        "# 手册\n\n## 实例\n\n示例：\n\n```md\n# 伪标题\n"
        "不应重分节。\n```\n\n真实正文。"
    )
    chunks = build_chunks(md, content_type="manual")
    assert chunks
    assert all(chunk.section_path == "手册 / 实例" for chunk in chunks)
    assert any("真实正文" in chunk.answer for chunk in chunks)


def test_dangling_marker_before_next_heading_is_rejected():
    import pytest

    md = "# 政策\n\n<!-- key-clause -->\n\n## 退款\n\n退货说明。"
    with pytest.raises(ValueError, match="key-clause"):
        build_chunks(md, content_type="policy")
