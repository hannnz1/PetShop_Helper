"""Sentence overlap must never begin inside a sentence."""

from app.kb import chunking


def test_overlap_uses_complete_trailing_sentence():
    chunks = ["前面很多内容。中间一句话。最后收尾句。", "下一块正文。"]
    assert chunking.apply_sentence_overlap(chunks, overlap=6) == [
        chunks[0], "最后收尾句。下一块正文。",
    ]


def test_overlap_does_not_slice_long_sentence_midway():
    sentence = "这是一个非常非常非常长的句子没有中间标点结尾才有。"
    out = chunking.apply_sentence_overlap([sentence, "新块。"], overlap=5)
    assert out[1] in (sentence + "新块。", "新块。")


def test_zero_overlap_leaves_chunks_unchanged():
    chunks = ["上一句。", "下一句。"]
    assert chunking.apply_sentence_overlap(chunks, overlap=0) == chunks


def test_empty_chunks_return_empty():
    assert chunking.apply_sentence_overlap([], overlap=5) == []


def test_consecutive_sentence_marks_stay_with_sentence():
    out = chunking.apply_sentence_overlap(["真的吗？！", "下一块。"], overlap=1)
    assert out[1] == "真的吗？！下一块。"


def test_closing_quote_stays_with_sentence():
    out = chunking.apply_sentence_overlap(["客服说‘可以退。’", "下一块。"], overlap=1)
    assert out[1] == "客服说‘可以退。’下一块。"
