import json

from scripts.ch10.export_onnx import export_options, export_and_verify, verify_predictions, prepare_serving_bundle


def test_export_options_fix_opset_and_dynamic_batch_sequence():
    options = export_options()
    assert options["opset_version"] == 17
    assert options["dynamo"] is False
    assert options["dynamic_axes"]["input_ids"] == {0: "batch", 1: "sequence"}
    assert options["dynamic_axes"]["logits"] == {0: "batch"}


def test_complete_threshold_matrix_mismatch_fails():
    from app.core.taxonomy import NUM_CLASSES

    assert verify_predictions([[0.7, 0.1] + [0.0] * (NUM_CLASSES - 2),
                               [0.2, 0.8] + [0.0] * (NUM_CLASSES - 2)],
                              [[0.7, 0.1] + [0.0] * (NUM_CLASSES - 2),
                               [0.49, 0.2] + [0.0] * (NUM_CLASSES - 2)], 0.5) == 1


def test_all_below_threshold_argmax_disagreement_fails():
    from app.core.taxonomy import NUM_CLASSES

    torch_scores = [[0.49, 0.48] + [0.0] * (NUM_CLASSES - 2)]
    onnx_scores = [[0.48, 0.49] + [0.0] * (NUM_CLASSES - 2)]
    assert verify_predictions(torch_scores, onnx_scores, 0.5) == 1


def test_missing_model_writes_pending_report(tmp_path):
    report = export_and_verify(tmp_path / "missing", tmp_path / "test.jsonl", tmp_path / "out")
    assert report.status == "pending_compute"
    saved = json.loads((tmp_path / "out" / "export_report.json").read_text(encoding="utf-8"))
    assert saved["status"] == "pending_compute"


def test_serving_bundle_copies_threshold_and_tokenizer(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "threshold.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "onnx"
    prepare_serving_bundle(model, target)
    assert (target / "threshold.json").exists()
    assert (target / "tokenizer.json").exists()
