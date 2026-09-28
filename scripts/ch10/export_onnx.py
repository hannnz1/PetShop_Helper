"""Export a trained checkpoint and compare all held-out thresholded labels."""

import argparse
import json
import hashlib
import shutil
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from app.core.taxonomy import LABEL2ID
from scripts.ch10.build_dataset import _hash_json
from scripts.ch10.inference_lib import apply_threshold


@dataclass(frozen=True)
class ExportReport:
    status: str
    checked: int = 0
    mismatches: int = 0
    reason: str = ""
    metadata: dict = field(default_factory=dict)


def export_options() -> dict:
    return {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        "opset_version": 17,
        "dynamo": False,
    }


def verify_predictions(torch_probs, onnx_probs, threshold: float) -> int:
    if len(torch_probs) != len(onnx_probs):
        raise ValueError("prediction row count mismatch")
    import numpy as np

    left = apply_threshold(np.asarray(torch_probs, dtype=float), threshold)
    right = apply_threshold(np.asarray(onnx_probs, dtype=float), threshold)
    return int(np.any(left != right, axis=1).sum())


def _report(output_dir: Path, result: ExportReport) -> ExportReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "export_report.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def prepare_serving_bundle(model_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("threshold.json", "tokenizer.json"):
        shutil.copyfile(model_dir / filename, output_dir / filename)


def _export_and_verify(model_dir: Path, test_path: Path, output_dir: Path) -> ExportReport:
    if not (model_dir / "threshold.json").exists():
        return _report(output_dir, ExportReport("pending_compute", reason="trained model missing"))
    if not test_path.exists() or not test_path.read_text(encoding="utf-8").strip():
        return _report(output_dir, ExportReport("pending_data", reason="held-out test missing"))
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return _report(output_dir, ExportReport("pending_compute", reason="optional ML dependencies missing"))

    metadata = json.loads((model_dir / "threshold.json").read_text(encoding="utf-8"))
    if metadata.get("label_order") != list(LABEL2ID):
        raise ValueError("trained label order differs from taxonomy")
    threshold = float(metadata["threshold"])
    samples = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if metadata.get("split_hashes", {}).get("test") != _hash_json(samples):
        return _report(output_dir, ExportReport("failed", reason="test split hash mismatch"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    encoded_example = tokenizer([samples[0]["text"]], padding=True, truncation=True,
                                max_length=128, return_tensors="pt")
    onnx_path = output_dir / "model.onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(LogitsOnly(model),
                      (encoded_example["input_ids"], encoded_example["attention_mask"]),
                      str(onnx_path), **export_options())
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # A newly loaded checkpoint is the reference; the exporter wrapper is not reused.
    fresh = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    fresh.eval()
    torch_probs = []
    onnx_probs = []
    for start in range(0, len(samples), 32):
        texts = [row["text"] for row in samples[start:start + 32]]
        batch = tokenizer(texts, padding=True, truncation=True, max_length=128,
                          return_tensors="pt")
        inputs = {name: batch[name] for name in ("input_ids", "attention_mask")}
        with torch.no_grad():
            logits = fresh(**inputs).logits
            torch_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        ort_logits = session.run(None, {name: np.ascontiguousarray(tensor.numpy())
                                        for name, tensor in inputs.items()})[0]
        onnx_probs.extend((1 / (1 + np.exp(-ort_logits))).tolist())
    mismatches = verify_predictions(torch_probs, onnx_probs, threshold)
    if mismatches == 0:
        prepare_serving_bundle(model_dir, output_dir)
    return _report(output_dir, ExportReport("passed" if mismatches == 0 else "failed",
                                            len(samples), mismatches,
                                            "" if mismatches == 0 else "thresholded label mismatch"))


def export_and_verify(model_dir: Path, test_path: Path, output_dir: Path) -> ExportReport:
    """Persist a failure report even when export or runtime loading raises."""
    try:
        result = _export_and_verify(model_dir, test_path, output_dir)
        if result.status == 'passed':
            from scripts.ch10.inference_lib import bundle_metadata
            from scripts.ch10.train import checkpoint_hash
            threshold = json.loads((output_dir/'threshold.json').read_text(encoding='utf-8'))['threshold']
            source_hash = checkpoint_hash(model_dir)
            if source_hash is None:
                raise ValueError('source checkpoint missing')
            meta = {**bundle_metadata(output_dir, threshold),
                    'test_hash': hashlib.sha256(test_path.read_bytes()).hexdigest(),
                    'checkpoint_hash': source_hash}
            result = replace(result, metadata=meta)
            _report(output_dir, result)
        return result
    except Exception as exc:
        return _report(output_dir, ExportReport("failed", reason=type(exc).__name__))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("data/ch10/model"))
    parser.add_argument("--test", type=Path, default=Path("data/ch10/dataset/test.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/ch10/onnx"))
    args = parser.parse_args()
    result = export_and_verify(args.model, args.test, args.out)
    print(f"status={result.status} checked={result.checked} mismatches={result.mismatches}")


if __name__ == "__main__":
    main()
