"""Build fixed multilabel splits before any training-only augmentation."""

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.core.taxonomy import LABEL2ID, terminology_table
from scripts.ch10.corpus_lib import desensitize, split_dataset


SUPPLEMENT = Path(__file__).with_name("supplement_sizefit.jsonl")


def fingerprint(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[\W_]+", "", folded)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _hash_json(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetBuild:
    train: tuple[dict, ...]
    val: tuple[dict, ...]
    test: tuple[dict, ...]
    manifest: dict


def build_dataset(
    samples: list[dict], augment_fn=None, *, seed: int = 42,
    supplement: list[dict] | None = None,
) -> DatasetBuild:
    """Preserve holdouts and reject any normalized cross-split duplicate."""
    if not samples:
        raise ValueError("reviewed corpus is empty")
    for row in samples:
        if not (row.get("reviewed") is True or row.get("origin") == "synthetic_reviewed"):
            raise ValueError("human review required before dataset build")
        labels = row.get("labels")
        if not labels or any(label not in LABEL2ID for label in labels):
            raise ValueError("unknown or empty label")
        if desensitize(row["text"]) != row["text"]:
            raise ValueError("source text contains private identifier")
    originals = [dict(row) for row in samples]
    original_fingerprints = [fingerprint(row["text"]) for row in originals]
    if len(set(original_fingerprints)) != len(original_fingerprints):
        raise ValueError("duplicate original text across dataset")
    train, val, test = split_dataset(originals, seed=seed)
    seen = set(original_fingerprints)
    if augment_fn is not None:
        for row in list(train):
            variant = augment_fn(row)
            if not isinstance(variant, str) or not variant.strip():
                continue
            variant = variant.strip()
            if desensitize(variant) != variant:
                continue
            key = fingerprint(variant)
            if key in seen:
                continue
            seen.add(key)
            train.append({"text": variant, "labels": list(row["labels"]),
                          "origin": "augmented", "reviewed": True})
    for row in supplement or []:
        if not row.get("labels") or any(label not in LABEL2ID for label in row["labels"]):
            raise ValueError("invalid supplement label")
        text = row["text"].strip()
        if desensitize(text) != text:
            raise ValueError("supplement contains private identifier")
        key = fingerprint(text)
        if key in seen:
            continue
        seen.add(key)
        train.append({"text": text, "labels": list(row["labels"]),
                      "origin": "supplement", "reviewed": True})
    train_keys = {fingerprint(row["text"]) for row in train}
    holdout_keys = {fingerprint(row["text"]) for row in val + test}
    if train_keys & holdout_keys or len(holdout_keys) != len(val) + len(test):
        raise ValueError("train/validation/test leakage")
    class_counts = {label: sum(label in row["labels"] for row in originals)
                    for label in LABEL2ID}
    ready = bool(val and test) and all(count >= 100 for count in class_counts.values())
    manifest = {
        "seed": seed,
        "taxonomy_hash": hashlib.sha256(terminology_table().encode("utf-8")).hexdigest(),
        "corpus_hash": _hash_json([{ "text": row["text"], "labels": row["labels"]} for row in originals]),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "class_counts_original": class_counts,
        "status": "ready" if ready else "pending_data",
    }
    return DatasetBuild(tuple(train), tuple(val), tuple(test), manifest)


def write_dataset(build: DatasetBuild, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        rows = getattr(build, name)
        (directory / f"{name}.jsonl").write_text(
            "\n".join(json.dumps({"text": row["text"], "labels": row["labels"]},
                                 ensure_ascii=False) for row in rows), encoding="utf-8",
        )
    (directory / "manifest.json").write_text(
        json.dumps(build.manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/ch10/corpus_labeled.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/ch10/dataset"))
    parser.add_argument("--include-supplement", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    supplement = ([json.loads(line) for line in SUPPLEMENT.read_text(encoding="utf-8").splitlines()
                   if line.strip()] if args.include_supplement else None)
    build = build_dataset(rows, supplement=supplement)
    write_dataset(build, args.out)
    print(f"status={build.manifest['status']} train={len(build.train)} "
          f"val={len(build.val)} test={len(build.test)}")


if __name__ == "__main__":
    main()
