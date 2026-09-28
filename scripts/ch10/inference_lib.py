"""Shared ONNX inference semantics for HTTP and offline evaluation."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.taxonomy import NUM_CLASSES, TOPIC_NAMES, terminology_table


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(probs, dtype=float)
    if values.ndim != 2 or values.shape[1] != NUM_CLASSES:
        raise ValueError("expected [batch, 17] score matrix")
    selected = values >= threshold
    for index in np.where(~selected.any(axis=1))[0]:
        selected[index, int(values[index].argmax())] = True
    return selected.astype(int)


@dataclass
class Runtime:
    session: object
    tokenizer: object
    threshold: float
    metadata: dict = field(default_factory=dict)

    def classify(self, texts: list[str]) -> list[dict]:
        if not texts:
            raise ValueError("at least one text required")
        result = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            encoded = self.tokenizer.encode_batch(batch)
            feed = {
                "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
            }
            logits = np.asarray(self.session.run(None, feed)[0], dtype=float)
            scores = 1 / (1 + np.exp(-logits))
            labels = apply_threshold(scores, self.threshold)
            for score_row, label_row in zip(scores, labels, strict=True):
                result.append({
                    "labels": [name for name, active in zip(TOPIC_NAMES, label_row, strict=True)
                               if active],
                    "scores": {name: round(float(score), 6)
                               for name, score in zip(TOPIC_NAMES, score_row, strict=True)},
                })
        return result


def bundle_metadata(model_dir: Path, threshold: float) -> dict:
    digest = hashlib.sha256()
    for name in ('model.onnx', 'tokenizer.json', 'threshold.json'):
        digest.update(name.encode())
        with (model_dir / name).open('rb') as stream:
            content_hash = hashlib.file_digest(stream, 'sha256').digest()
        digest.update(content_hash)
    return {'model_version': digest.hexdigest(),
            'taxonomy_hash': hashlib.sha256(terminology_table().encode('utf-8')).hexdigest(),
            'threshold': threshold}


def load_runtime(model_dir: Path) -> Runtime:
    metadata = json.loads((model_dir / "threshold.json").read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(terminology_table().encode("utf-8")).hexdigest()
    if metadata.get("taxonomy_hash") != expected_hash or metadata.get("label_order") != list(TOPIC_NAMES):
        raise ValueError("taxonomy hash or label order mismatch")
    if not 0 < float(metadata["threshold"]) < 1:
        raise ValueError("invalid threshold")
    onnx_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"
    if not onnx_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError("ONNX model or tokenizer missing")
    report = json.loads((model_dir / "export_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("ONNX export agreement has not passed")
    runtime_metadata = bundle_metadata(model_dir, float(metadata['threshold']))
    exported = report.get('metadata', {})
    if any(exported.get(key) != value for key, value in runtime_metadata.items()):
        raise ValueError('ONNX export report version differs from bundle; re-export and verify')
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=128)
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise ValueError("tokenizer has no [PAD] token")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    threshold = float(metadata['threshold'])
    return Runtime(session, tokenizer, threshold, runtime_metadata)
