import pytest


def test_review_import_only_approved_ids_and_preserves_lineage():
    from scripts.ch10.review_corpus import apply_reviews

    rows = [{"id": 17, "text": "退款多久", "labels": ["退换货"], "origin": "pool"},
            {"review_id": "sim-1", "text": "猫窝尺寸", "labels": ["尺码"], "origin": "simulated"}]
    decisions = [{"review_id": "pool-17", "approved": True, "labels": ["退换货"]},
                 {"review_id": "sim-1", "approved": False}]
    approved = apply_reviews(rows, decisions)
    assert approved == [{"id": 17, "review_id": "pool-17", "text": "退款多久",
                         "labels": ["退换货"], "origin": "pool", "reviewed": True}]
    with pytest.raises(ValueError, match="unknown|duplicate"):
        apply_reviews(rows, [{"review_id": "missing", "approved": True, "labels": ["退换货"]}])
