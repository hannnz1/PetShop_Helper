from scripts.ch10.corpus_lib import dedupe, desensitize, split_dataset


def test_desensitize_masks_identifiers_preserving_product_model():
    raw = "订单202601180001234567，电话13812345678，邮箱a@example.com，微信 cat_lover2026，MH-LP100"
    masked = desensitize(raw)
    for secret in ("202601180001234567", "13812345678", "a@example.com", "cat_lover2026"):
        assert secret not in masked
    assert "MH-LP100" in masked


def test_desensitize_masks_numeric_qq_before_audit_or_model():
    assert "123456789" not in desensitize("QQ号: 123456789，咨询退货")


def test_dedupe_keeps_first_source_and_drops_blank():
    rows = [{"text": " 退货 ", "origin": "pool"}, {"text": "退货", "origin": "simulated"},
            {"text": "  ", "origin": "pool"}]
    assert dedupe(rows) == [{"text": "退货", "origin": "pool"}]


def test_split_is_reproducible_and_rare_layers_stay_in_train():
    rows = ([{"text": f"退货{i}", "labels": ["退换货"]} for i in range(40)]
            + [{"text": f"物流{i}", "labels": ["物流"]} for i in range(40)]
            + [{"text": "稀有1", "labels": ["发票", "价保"]},
               {"text": "稀有2", "labels": ["发票", "价保"]}])
    first = split_dataset(rows, seed=42)
    second = split_dataset(rows, seed=42)
    assert first == second
    train, val, test = first
    assert {r["text"] for r in train + val + test} == {r["text"] for r in rows}
    assert {"稀有1", "稀有2"} <= {r["text"] for r in train}
    assert all(not {"稀有1", "稀有2"} & {r["text"] for r in part} for part in (val, test))
