"""Offline checks for the Ch03 annotated evaluation rules."""

import json
from pathlib import Path

from scripts.eval_mining import check_sample as check_mining
from scripts.eval_retrieval import check_sample as check_retrieval


DATA = Path(__file__).resolve().parent / "data"


def test_retrieval_checks_answer_facts_not_only_question():
    sample = {
        "query": "快递费怎么收",
        "expect_question_contains": "运费",
        "expect_answer_contains_all": ["99", "10", "偏远地区"],
    }
    assert check_retrieval(sample, {"question": "运费怎么算", "answer": "满99元包邮，未满收取10元运费，偏远地区另计。"})[0]
    assert not check_retrieval(sample, {"question": "运费怎么算", "answer": "运费以页面显示为准。"})[0]
    assert not check_retrieval(sample, None)[0]


def test_mining_checks_answer_fidelity_and_privacy():
    sample = {
        "expect_pairs": [{
            "question_contains": ["运费"],
            "answer_contains_all": ["99", "10"],
        }],
        "forbid_anywhere": ["MH123", "保证赔偿"],
    }
    good = [{"question": "运费怎么算", "answer": "满99元包邮，未满收10元。"}]
    assert check_mining(sample, good)[0]
    assert not check_mining(sample, [{"question": "运费怎么算", "answer": "满99元包邮，保证赔偿。"}])[0]
    assert not check_mining(sample, [{"question": "MH123运费怎么算", "answer": "满99元包邮，未满收10元。"}])[0]


def test_mining_empty_means_no_extraction():
    sample = {"expect_empty": True}
    assert check_mining(sample, [])[0]
    assert not check_mining(sample, [{"question": "订单到哪了", "answer": "广州分拨中心"}])[0]


def test_annotated_fee_policy_rejects_reversed_relationship_and_extra_promise():
    mining_sample = next(s for s in json.loads((DATA / "mining_samples.json").read_text(encoding="utf-8")) if s["id"] == "fee-preserve-source-amounts")
    retrieval_sample = next(s for s in json.loads((DATA / "retrieval_samples.json").read_text(encoding="utf-8")) if s["id"] == "shipping-fee-paraphrase")

    def pair(answer):
        return {"question": "运费怎么算", "answer": answer}

    faithful = "订单金额达到99元免运费；不足99元需付10元邮费，偏远地区另计。"
    reversed_policy = "订单满99元收10元运费，未满99元包邮，偏远地区另计。"
    unsupported = "满99元包邮，未满收10元运费，偏远地区另计，并保证明天送达。"

    assert check_mining(mining_sample, [pair(faithful)])[0]
    assert check_retrieval(retrieval_sample, pair(faithful))[0]
    assert not check_mining(mining_sample, [pair(reversed_policy)])[0]
    assert not check_retrieval(retrieval_sample, pair(reversed_policy))[0]
    assert not check_mining(mining_sample, [pair(unsupported)])[0]
    assert not check_retrieval(retrieval_sample, pair(unsupported))[0]
