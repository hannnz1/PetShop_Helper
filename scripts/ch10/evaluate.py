"""Evaluate held-out multilabel data with severity-aware redlines."""

import argparse
import json
from pathlib import Path

import numpy as np

from app.core.taxonomy import NUM_CLASSES, SEVERITY, TOPIC_NAMES
from scripts.ch10.inference_lib import load_runtime


def _f1(gold: np.ndarray, predicted: np.ndarray) -> float:
    tp = int(np.logical_and(gold, predicted).sum())
    fp = int(np.logical_and(~gold, predicted).sum())
    fn = int(np.logical_and(gold, ~predicted).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def evaluate_matrix(gold, predicted) -> dict:
    truth = np.asarray(gold, dtype=bool)
    guess = np.asarray(predicted, dtype=bool)
    if truth.shape != guess.shape or truth.ndim != 2 or truth.shape[1] != NUM_CLASSES:
        raise ValueError("gold and predicted must have matching 17-class shape")
    per_class = {}
    redlines = []
    for i, name in enumerate(TOPIC_NAMES):
        support = int(truth[:, i].sum())
        score = _f1(truth[:, i], guess[:, i])
        per_class[name] = {"f1": round(score, 6), "support": support,
                           "severity": SEVERITY[name]}
        minimum = {"严": 0.9, "中": 0.8}.get(SEVERITY[name])
        if minimum is not None and (support == 0 or score < minimum):
            redlines.append(name)
    scores = [value["f1"] for value in per_class.values()]
    return {"status": "passed" if not redlines else "failed",
            "micro_f1": round(_f1(truth, guess), 6),
            "macro_f1": round(sum(scores) / NUM_CLASSES, 6),
            "per_class": per_class, "redline_classes": redlines}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("data/ch10/onnx"))
    parser.add_argument("--test", type=Path, default=Path("data/ch10/dataset/test.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/ch10/evaluation.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.test.read_text(encoding="utf-8").splitlines() if line.strip()]
    runtime = load_runtime(args.model)
    predictions = runtime.classify([row["text"] for row in rows])
    report = evaluate_matrix([[int(name in row["labels"]) for name in TOPIC_NAMES] for row in rows],
                             [[int(name in row["labels"]) for name in TOPIC_NAMES] for row in predictions])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status={report['status']} micro_f1={report['micro_f1']}")


if __name__ == "__main__":
    main()
