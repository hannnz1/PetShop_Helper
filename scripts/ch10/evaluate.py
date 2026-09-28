"""Evaluate held-out multilabel data with severity-aware redlines."""

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np

from app.core.taxonomy import NUM_CLASSES, SEVERITY, TOPIC_NAMES, terminology_table
from scripts.ch10.inference_lib import load_runtime
from scripts.ch10.corpus_lib import desensitize


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
        tp = int(np.logical_and(truth[:, i], guess[:, i]).sum())
        fp = int(np.logical_and(~truth[:, i], guess[:, i]).sum())
        fn = int(np.logical_and(truth[:, i], ~guess[:, i]).sum())
        tn = int(np.logical_and(~truth[:, i], ~guess[:, i]).sum())
        minimum = {"严": 0.9, "中": 0.8}.get(SEVERITY[name])
        per_class[name] = {"f1": round(score, 6), "support": support,
                           "severity": SEVERITY[name], "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                           "precision": round(tp / (tp + fp), 6) if tp + fp else 0.0,
                           "recall": round(tp / (tp + fn), 6) if tp + fn else 0.0,
                           "passed": None if minimum is None else bool(support and score >= minimum)}
        if minimum is not None and (support == 0 or score < minimum):
            redlines.append(name)
    scores = [value["f1"] for value in per_class.values()]
    total_tp = sum(value['tp'] for value in per_class.values())
    total_fp = sum(value['fp'] for value in per_class.values())
    total_fn = sum(value['fn'] for value in per_class.values())
    return {"status": "passed" if not redlines else "failed",
            "micro_precision": round(total_tp / (total_tp + total_fp), 6) if total_tp + total_fp else 0.0,
            "micro_recall": round(total_tp / (total_tp + total_fn), 6) if total_tp + total_fn else 0.0,
            "macro_precision": round(sum(value['precision'] for value in per_class.values()) / NUM_CLASSES, 6),
            "macro_recall": round(sum(value['recall'] for value in per_class.values()) / NUM_CLASSES, 6),
            "micro_f1": round(_f1(truth, guess), 6),
            "macro_f1": round(sum(scores) / NUM_CLASSES, 6),
            "per_class": per_class, "redline_classes": redlines}


def build_error_samples(rows: list[dict], predicted: list[dict]) -> list[dict]:
    errors = []
    for index, (row, prediction) in enumerate(zip(rows, predicted, strict=True)):
        gold, guess = set(row['labels']), set(prediction['labels'])
        missed = [name for name in TOPIC_NAMES if name in gold - guess]
        extra = [name for name in TOPIC_NAMES if name in guess - gold]
        if missed or extra:
            errors.append({'index': index, 'review_id': row.get('review_id'),
                           'text': desensitize(row['text']), 'missed': missed, 'extra': extra,
                           'kind': 'mixed' if missed and extra else 'missed' if missed else 'extra'})
    return errors


def write_evaluation(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 分类质量报告', '', f"状态：{report['status']}",
             f"micro F1：{report['micro_f1']}；macro F1：{report['macro_f1']}", '',
             f"micro P/R：{report.get('micro_precision')} / {report.get('micro_recall')}；macro P/R：{report.get('macro_precision')} / {report.get('macro_recall')}", '',
             '元数据：`' + json.dumps(report.get('metadata', {}), ensure_ascii=False) + '`', '',
             '| 类别 | Precision | Recall | F1 | TN | FP | FN | TP | Support | Passed |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for name, row in report['per_class'].items():
        lines.append('| ' + ' | '.join([name, *(str(row[key]) for key in
                     ('precision', 'recall', 'f1', 'tn', 'fp', 'fn', 'tp', 'support', 'passed'))]) + ' |')
    lines.extend(['', '## 错例', '', '```json',
                  json.dumps(report.get('errors', []), ensure_ascii=False, indent=2), '```', ''])
    output.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')


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
    report['errors'] = build_error_samples(rows, predictions)
    report['metadata'] = {
        **getattr(runtime, 'metadata', {}), 'threshold': runtime.threshold,
        'taxonomy_hash': hashlib.sha256(terminology_table().encode('utf-8')).hexdigest(),
        'test_hash': hashlib.sha256(args.test.read_bytes()).hexdigest(),
    }
    write_evaluation(report, args.out)
    print(f"status={report['status']} micro_f1={report['micro_f1']}")


if __name__ == "__main__":
    main()
