"""Replay saved validation probabilities against the fixed threshold grid."""

import argparse
import json
from pathlib import Path

from app.core.taxonomy import TOPIC_NAMES
from scripts.ch10.inference_lib import load_runtime
from scripts.ch10.train import scan_threshold


def replay(rows: list[dict], scores: list[dict[str, float]], saved_threshold: float) -> dict:
    if len(rows) != len(scores) or not rows:
        raise ValueError("validation rows and scores must align")
    gold = [[float(label in row["labels"]) for label in TOPIC_NAMES] for row in rows]
    probs = [[float(item[name]) for name in TOPIC_NAMES] for item in scores]
    best = scan_threshold(probs, gold)
    return {"saved_threshold": saved_threshold, "replayed_threshold": best.threshold,
            "val_micro_f1": best.micro_f1,
            "status": "passed" if abs(saved_threshold - best.threshold) < 1e-9 else "failed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("data/ch10/onnx"))
    parser.add_argument("--val", type=Path, default=Path("data/ch10/dataset/val.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/ch10/threshold_replay.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.val.read_text(encoding="utf-8").splitlines() if line.strip()]
    runtime = load_runtime(args.model)
    scores = [item["scores"] for item in runtime.classify([row["text"] for row in rows])]
    report = replay(rows, scores, runtime.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status={report['status']} replayed_threshold={report['replayed_threshold']}")


if __name__ == "__main__":
    main()
