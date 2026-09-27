import json
from pathlib import Path

import pytest

from app.core.taxonomy import LABEL2ID, NUM_CLASSES


def test_encode_has_float_multihot_labels():
    from scripts.ch10.train import encode

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1, 2] for _ in texts],
                    "attention_mask": [[1, 1] for _ in texts]}

    encoded = encode([{"text": "退款与物流", "labels": ["退换货", "物流"]}], Tokenizer())
    labels = encoded[0]["labels"]
    assert len(labels) == NUM_CLASSES
    assert labels[LABEL2ID["退换货"]] == labels[LABEL2ID["物流"]] == 1.0
    assert all(isinstance(value, float) for value in labels)


def test_threshold_grid_chooses_first_tie():
    from scripts.ch10.train import scan_threshold

    result = scan_threshold([[0.95] + [0.05] * (NUM_CLASSES - 1)],
                            [[1.0] + [0.0] * (NUM_CLASSES - 1)])
    assert result.threshold == 0.30
    assert result.micro_f1 == 1.0


def test_training_refuses_unready_dataset_without_importing_heavy_libraries(tmp_path: Path):
    from scripts.ch10.train import train_from_dataset

    (tmp_path / "manifest.json").write_text(json.dumps({"status": "pending_data"}), encoding="utf-8")
    assert train_from_dataset(tmp_path, tmp_path / "model").status == "pending_data"


def test_training_refuses_taxonomy_hash_mismatch(tmp_path: Path):
    from scripts.ch10.train import train_from_dataset

    (tmp_path / "manifest.json").write_text(json.dumps({"status": "ready", "taxonomy_hash": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="taxonomy"):
        train_from_dataset(tmp_path, tmp_path / "model")


def test_training_rejects_tampered_split(tmp_path: Path):
    from scripts.ch10.build_dataset import build_dataset, write_dataset
    from scripts.ch10.train import train_from_dataset

    rows = [{"text": f"问题{i}", "labels": ["物流"], "reviewed": True} for i in range(10)]
    build = build_dataset(rows)
    write_dataset(build, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "ready"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "train.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="split|hash"):
        train_from_dataset(tmp_path, tmp_path / "model")
