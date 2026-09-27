import pytest


def _rows():
    return ([{"text": f"退货问题{i}", "labels": ["退换货"], "reviewed": True} for i in range(40)]
            + [{"text": f"物流问题{i}", "labels": ["物流"], "reviewed": True} for i in range(40)])


def test_dataset_split_fixed_seed_and_no_augmentation_leakage():
    from scripts.ch10.build_dataset import build_dataset, fingerprint

    rows = _rows()
    augmented_inputs = []

    def augment(sample):
        augmented_inputs.append(sample["text"])
        return sample["text"] + " 变体"

    result = build_dataset(rows, augment_fn=augment, seed=42)
    again = build_dataset(rows, augment_fn=augment, seed=42)
    assert result.train == again.train and result.val == again.val and result.test == again.test
    assert len(result.val) > 0 and len(result.test) > 0
    assert set(augmented_inputs) <= {row["text"] for row in result.train}
    train_hashes = {fingerprint(row["text"]) for row in result.train}
    holdout_hashes = {fingerprint(row["text"]) for row in result.val + result.test}
    assert not train_hashes & holdout_hashes
    assert result.manifest["seed"] == 42 and result.manifest["taxonomy_hash"]


def test_augmentation_collision_and_supplement_only_train():
    from scripts.ch10.build_dataset import build_dataset

    rows = _rows()
    result = build_dataset(rows, augment_fn=lambda sample: rows[0]["text"],
                           supplement=[{"text": "买大了想退", "labels": ["尺码", "退换货"]}])
    assert len([row for row in result.train if row.get("origin") == "augmented"]) == 0
    assert any(row["text"] == "买大了想退" for row in result.train)
    assert not any(row["text"] == "买大了想退" for row in result.val + result.test)


def test_unreviewed_or_unknown_labels_are_refused():
    from scripts.ch10.build_dataset import build_dataset

    with pytest.raises(ValueError, match="review"):
        build_dataset([{"text": "真实用户问题", "labels": ["退换货"]}])
    with pytest.raises(ValueError, match="label"):
        build_dataset([{"text": "合成样本", "labels": ["虚构类"], "reviewed": True}])


def test_tiny_synthetic_holdouts_do_not_claim_training_readiness():
    from scripts.ch10.build_dataset import build_dataset

    rows = [{"text": f"退货样本{i}", "labels": ["退换货"], "reviewed": True}
            for i in range(30)]
    result = build_dataset(rows)
    assert result.val and result.test
    assert result.manifest["status"] == "pending_data"


def test_split_hashes_and_review_lineage_are_persisted(tmp_path):
    import json
    from scripts.ch10.build_dataset import build_dataset, write_dataset

    result = build_dataset(_rows())
    write_dataset(result, tmp_path)
    persisted = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert persisted["reviewed"] is True
    assert set(result.manifest["split_hashes"]) == {"train", "val", "test"}
