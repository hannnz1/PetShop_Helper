import pytest

from app.kb import chunking


TABLE = "\n".join(
    [
        "| 商品 | 价格 |",
        "| --- | --- |",
        "| A | 1 |",
        "| B | 2 |",
        "| C | 3 |",
    ]
)


def test_is_table_block() -> None:
    assert chunking.is_table_block(TABLE)
    assert not chunking.is_table_block("普通段落文字。")
    assert not chunking.is_table_block("| 商品 | 价格 |\n| A | 1 |")


def test_split_table_replicates_header() -> None:
    out = chunking.split_table_rows(TABLE, max_rows=2)
    assert out == [
        "| 商品 | 价格 |\n| --- | --- |\n| A | 1 |\n| B | 2 |",
        "| 商品 | 价格 |\n| --- | --- |\n| C | 3 |",
    ]
    assert all(block.startswith("| 商品 | 价格 |\n| --- | --- |") for block in out)


def test_split_table_small_stays_whole() -> None:
    assert chunking.split_table_rows(TABLE, max_rows=10) == [TABLE]


@pytest.mark.parametrize("max_rows", [0, -1])
def test_split_table_rejects_nonpositive_max_rows(max_rows: int) -> None:
    with pytest.raises(ValueError, match="max_rows"):
        chunking.split_table_rows(TABLE, max_rows=max_rows)


def test_split_table_rejects_non_table() -> None:
    with pytest.raises(ValueError, match="table"):
        chunking.split_table_rows("普通段落文字。", max_rows=2)


@pytest.mark.parametrize(
    "malformed",
    [
        TABLE + "\n普通段落文字。",
        "| 商品 | 价格 |\n| --- | --- |\n| A |",
        "| 商品 | 价格 |\n| --- |\n| A | 1 |",
        "| 商品 | 价格 |\n| --- | --- |\n| A | 1 | 额外列 |",
    ],
)
def test_malformed_table_rows_are_rejected(malformed: str) -> None:
    assert not chunking.is_table_block(malformed)
    with pytest.raises(ValueError, match="table"):
        chunking.split_table_rows(malformed, max_rows=2)
