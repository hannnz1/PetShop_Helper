"""Build fixed multilabel splits before any training-only augmentation."""

import argparse
import asyncio
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
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
                          "origin": "augmented", "reviewed": False,
                          "review_id": 'aug-' + key,
                          "parent_review_id": row.get('review_id'),
                          "parent_text_hash": fingerprint(row['text'])})
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
        "corpus_hash": _hash_json(originals),
        "split_hashes": {name: _hash_json(rows) for name, rows in
                         (("train", train), ("val", val), ("test", test))},
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "class_counts_original": class_counts,
        "status": 'pending_review' if any(row.get('reviewed') is not True for row in train) else "ready" if ready else "pending_data",
    }
    return DatasetBuild(tuple(train), tuple(val), tuple(test), manifest)


def review_augmentation(build: DatasetBuild, decisions: list[dict]) -> DatasetBuild:
    """Apply human decisions to train only; preserve the fixed holdout objects."""
    augmented = {row['review_id'] for row in build.train if row.get('origin') == 'augmented'}
    selected = {}
    for row in decisions:
        key = row.get('review_id')
        if key not in augmented or key in selected or type(row.get('approved')) is not bool:
            raise ValueError('invalid or duplicate augmentation decision')
        selected[key] = row['approved']
    train = tuple({**row, 'reviewed': True} if selected.get(row.get('review_id')) is True else row
                  for row in build.train if selected.get(row.get('review_id')) is not False)
    manifest = json.loads(json.dumps(build.manifest))
    pending = any(row.get('reviewed') is not True for row in train)
    ready = bool(build.val and build.test) and all(value >= 100 for value in manifest['class_counts_original'].values())
    manifest['status'] = 'pending_review' if pending else 'ready' if ready else 'pending_data'
    manifest['split_hashes']['train'] = _hash_json(train)
    manifest['counts']['train'] = len(train)
    manifest.setdefault('augmentation', {})['semantic_review'] = 'pending' if pending else 'complete'
    return replace(build, train=train, manifest=manifest)


def write_dataset(build: DatasetBuild, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        rows = getattr(build, name)
        (directory / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
            encoding="utf-8",
        )
    (directory / "manifest.json").write_text(
        json.dumps(build.manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/ch10/corpus_labeled.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/ch10/dataset"))
    parser.add_argument("--include-supplement", action="store_true")
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--augmentation-cache', type=Path, default=Path('data/ch10/augmentation-cache.json'))
    parser.add_argument('--augmentation-decisions', type=Path)
    parser.add_argument('--allow-external-real-text', action='store_true')
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    supplement = ([json.loads(line) for line in SUPPLEMENT.read_text(encoding="utf-8").splitlines()
                   if line.strip()] if args.include_supplement else None)
    augmentation = None
    prepared = None
    if args.augment:
        from scripts.ch10.text_jobs import external_allowed, prepare_augmentation
        if not external_allowed(args.allow_external_real_text):
            print('status=pending_upstream augmentation requires explicit real-text permission')
            return
        from app.core.llm import get_chat_model
        base = build_dataset(rows)
        prepared = asyncio.run(prepare_augmentation(list(base.train), get_chat_model(temperature=0),
                               args.augmentation_cache, allow_external_real_text=args.allow_external_real_text))
        augmentation = lambda row: prepared[fingerprint(row['text'])].get('text')
    build = build_dataset(rows, augment_fn=augmentation, supplement=supplement)
    if prepared is not None:
        counts = Counter(value['status'] for value in prepared.values())
        unavailable = counts['failed'] + counts['pending_upstream']
        build.manifest['augmentation'] = {
            'status': 'pending_upstream' if unavailable else 'complete',
            'counts': dict(counts),
            'accepted': sum(row.get('origin') == 'augmented' for row in build.train),
            'semantic_review': 'pending',
        }
        if unavailable and build.manifest['status'] == 'ready':
            build.manifest['status'] = 'partial'
    if args.augmentation_decisions:
        decisions = [json.loads(line) for line in args.augmentation_decisions.read_text(encoding='utf-8').splitlines() if line.strip()]
        build = review_augmentation(build, decisions)
        if prepared is not None and build.manifest['augmentation'].get('status') == 'pending_upstream' and build.manifest['status'] == 'ready':
            build.manifest['status'] = 'partial'
    write_dataset(build, args.out)
    print(f"status={build.manifest['status']} train={len(build.train)} "
          f"val={len(build.val)} test={len(build.test)}")


if __name__ == "__main__":
    main()
