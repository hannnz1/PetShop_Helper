import json

import numpy as np
import pytest

from app.core.taxonomy import LABEL2ID, NUM_CLASSES


def test_threshold_falls_back_to_highest_score():
    from scripts.ch10.inference_lib import apply_threshold

    scores = np.array([[0.1, 0.2] + [0.0] * (NUM_CLASSES - 2)])
    predicted = apply_threshold(scores, 0.5)
    assert predicted.shape == (1, NUM_CLASSES)
    assert predicted[0].sum() == 1 and predicted[0, 1] == 1


def test_load_runtime_rejects_label_order_mismatch_before_optional_import(tmp_path):
    from scripts.ch10.inference_lib import load_runtime

    (tmp_path / "threshold.json").write_text(json.dumps({
        "threshold": 0.5, "label_order": list(reversed(LABEL2ID)),
        "taxonomy_hash": "bad"}), encoding="utf-8")
    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="taxonomy|label order"):
        load_runtime(tmp_path)
