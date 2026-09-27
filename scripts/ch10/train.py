"""Offline full-parameter RoBERTa multilabel training, gated by reviewed data."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from app.core.taxonomy import ID2LABEL, LABEL2ID, NUM_CLASSES, terminology_table


BASE_MODEL = "hfl/chinese-roberta-wwm-ext"
GRID = tuple(round(0.30 + i * 0.05, 2) for i in range(9))


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    micro_f1: float


@dataclass(frozen=True)
class TrainReport:
    status: str
    reason: str = ""


def encode(samples: list[dict], tokenizer) -> list[dict]:
    if not samples:
        return []
    tokens = tokenizer([row["text"] for row in samples], truncation=True,
                       padding="max_length", max_length=128)
    result = []
    for i, row in enumerate(samples):
        vector = [0.0] * NUM_CLASSES
        for label in row["labels"]:
            vector[LABEL2ID[label]] = 1.0
        result.append({**{key: values[i] for key, values in tokens.items()},
                       "labels": vector})
    return result


def _micro_f1(gold, predicted) -> float:
    tp = fp = fn = 0
    for expected_row, predicted_row in zip(gold, predicted, strict=True):
        for expected, actual in zip(expected_row, predicted_row, strict=True):
            tp += bool(expected) and bool(actual)
            fp += not bool(expected) and bool(actual)
            fn += bool(expected) and not bool(actual)
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def scan_threshold(probs, gold) -> ThresholdResult:
    if len(probs) == 0 or len(probs) != len(gold):
        raise ValueError("validation predictions and labels required")
    best = ThresholdResult(GRID[0], -1.0)
    for threshold in GRID:
        predicted = [[float(prob) >= threshold for prob in row] for row in probs]
        score = _micro_f1(gold, predicted)
        if score > best.micro_f1:
            best = ThresholdResult(threshold, score)
    return best


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def train_from_dataset(dataset_dir: Path, model_dir: Path) -> TrainReport:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "ready":
        return TrainReport("pending_data", "reviewed per-class corpus below 100 or holdout missing")
    taxonomy_hash = hashlib.sha256(terminology_table().encode("utf-8")).hexdigest()
    if manifest.get("taxonomy_hash") != taxonomy_hash:
        raise ValueError("taxonomy hash mismatch")
    train_rows = _read_jsonl(dataset_dir / "train.jsonl")
    val_rows = _read_jsonl(dataset_dir / "val.jsonl")
    if not train_rows or not val_rows:
        return TrainReport("pending_data", "train/validation split missing")
    try:
        import numpy as np
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  Trainer, TrainerCallback, TrainingArguments,
                                  default_data_collator)
    except ImportError:
        return TrainReport("pending_compute", "install optional ml dependency group")

    class BestInMemory(TrainerCallback):
        def __init__(self, model):
            self.model = model
            self.best_score = -1.0
            self.best_state = None

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            score = (metrics or {}).get("eval_micro_f1", -1.0)
            if score > self.best_score:
                self.best_score = score
                self.best_state = {key: value.detach().to("cpu", copy=True)
                                   for key, value in self.model.state_dict().items()}

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_CLASSES, problem_type="multi_label_classification",
        id2label=ID2LABEL, label2id=LABEL2ID)
    train_data = encode(train_rows, tokenizer)
    val_data = encode(val_rows, tokenizer)

    def metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        return {"micro_f1": _micro_f1(labels, probs >= 0.5)}

    args = TrainingArguments(
        output_dir=str(model_dir / "logs"), eval_strategy="epoch", save_strategy="no",
        learning_rate=2e-5, per_device_train_batch_size=16,
        per_device_eval_batch_size=64, num_train_epochs=8, weight_decay=0.01,
        metric_for_best_model="micro_f1", greater_is_better=True,
        logging_steps=20, report_to="none")
    best = BestInMemory(model)
    trainer = Trainer(model=model, args=args, train_dataset=train_data,
                      eval_dataset=val_data, data_collator=default_data_collator,
                      compute_metrics=metrics, callbacks=[best])
    trainer.train()
    if best.best_state is None:
        return TrainReport("pending_compute", "no completed validation epoch")
    model.load_state_dict(best.best_state)
    predictions = trainer.predict(val_data).predictions
    probs = (1 / (1 + np.exp(-predictions))).tolist()
    threshold = scan_threshold(probs, [row["labels"] for row in val_data])
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    (model_dir / "threshold.json").write_text(json.dumps({
        "threshold": threshold.threshold, "val_micro_f1": threshold.micro_f1,
        "taxonomy_hash": taxonomy_hash, "dataset_hash": manifest["corpus_hash"],
        "label_order": list(LABEL2ID), "base_model": BASE_MODEL,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrainReport("trained")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/ch10/dataset"))
    parser.add_argument("--out", type=Path, default=Path("data/ch10/model"))
    args = parser.parse_args()
    result = train_from_dataset(args.dataset, args.out)
    print(f"status={result.status} reason={result.reason}")


if __name__ == "__main__":
    main()
