from dataclasses import dataclass

from app.kb import dedup


@dataclass
class Item:
    question: str


def test_normalize_strips_whitespace_punctuation_and_keeps_chinese():
    assert dedup.normalize_question(" 邮费,是多少? ") == "邮费是多少"
    assert dedup.normalize_question(" 退 货，怎么 办？\n") == "退货怎么办"
    assert dedup.normalize_question("  A_B-123！ ") == "ab123"


def test_dedupe_within_batch_and_against_existing():
    items = [Item("邮费是多少"), Item("邮费是多少?"), Item("怎么退货"), Item("运费怎么算")]
    kept, discarded = dedup.dedupe(items, existing_questions=["运费怎么算"])
    assert kept == [items[0], items[2]]
    assert discarded == [items[1], items[3]]


def test_dedupe_discards_empty_normalized_question_and_keeps_original_objects():
    items = [Item("？ _ "), Item("ABC?"), Item("a b c")]
    kept, discarded = dedup.dedupe(items, existing_questions=[])
    assert kept == [items[1]]
    assert discarded == [items[0], items[2]]
    assert kept[0] is items[1]


def test_normalize_unicode_equivalent_questions():
    assert dedup.normalize_question("cafe\u0301？") == dedup.normalize_question("café?")
    assert dedup.normalize_question("ＡＢＣ１２３？") == dedup.normalize_question("ABC123?")


def test_dedupe_rejects_unicode_equivalent_existing_question():
    item = Item("ＡＢＣ１２３？")
    assert dedup.dedupe([item], existing_questions=["ABC123?"]) == ([], [item])
