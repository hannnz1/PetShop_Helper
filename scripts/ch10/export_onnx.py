"""Export a trained checkpoint and compare all held-out thresholded labels."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.taxonomy import LABEL2ID


@dataclass(frozen=True)
class ExportReport:
    status: str
    checked: int = 0
    mismatches: int = 0
    reason: str = ""


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
    mismatches = 0
    for left, right in zip(torch_probs, onnx_probs, strict=True):
        if len(left) != len(right):
            raise ValueError("prediction label count mismatch")
        mismatches += any((float(a) >= threshold) != (float(b) >= threshold)
                          for a, b in zip(left, right, strict=True))
    return mismatches


def _report(output_dir: Path, result: ExportReport) -> ExportReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "export_report.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def export_and_verify(model_dir: Path, test_path: Path, output_dir: Path) -> ExportReport:
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
    return _report(output_dir, ExportReport("passed" if mismatches == 0 else "failed",
                                            len(samples), mismatches,
                                            "" if mismatches == 0 else "thresholded label mismatch"))


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
