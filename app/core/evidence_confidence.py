"""Deterministic rerank-only evidence scoring and provenance-bound calibration."""

import hashlib
import json
from math import isfinite
from pathlib import Path


KEY_TERMS = ('退款', '退货', '时效', '运费', '邮费', '费用', '保修', '赔偿', '期限', '包邮')
SIGNAL_VERSION = 'mewhelp-four-signals-v1'
ANSWERABLE = {'A_policy', 'B_model', 'C_colloquial'}
META_KEYS = {'dataset_hash', 'strategy', 'signal_version', 'retrieval_config_hash'}


def _number(value):
    if isinstance(value, bool):
        raise ValueError('score must be a finite number')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('score must be a finite number') from exc
    if not isfinite(number):
        raise ValueError('score must be a finite number')
    return number


def _clip(value):
    return max(0.0, min(1.0, value))


def compute_evidence_confidence(hits: list[dict]) -> dict:
    # Generic score fields may contain dense/BM25/RRF values: never use them.
    scores = [_number(hit.get('rerank_score')) for hit in hits]
    top1 = scores[0] if scores else 0.0
    margin = top1 - scores[1] if len(scores) > 1 else top1
    count = sum(score >= .3 for score in scores)
    key_hit = any(any(term in f"{hit.get('question', '')}{hit.get('answer', '')}"
                      for term in KEY_TERMS) for hit in hits[:3])
    score = .5 * _clip(top1) + .2 * min(count, 3) / 3 + .2 * _clip(margin) + .1 * key_hit
    return {'score': round(_clip(score), 4), 'signals': {
        'top1_score': top1, 'valid_count': count,
        'margin': round(margin, 4), 'key_clause_hit': key_hit,
    }}


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def retrieval_fingerprint(settings) -> str:
    # Include only public retrieval configuration, never API keys/credentials.
    fields = ('embed_model', 'rerank_model', 'recall_top_k', 'rerank_top_k',
              'chat_model', 'chat_base_url', 'structured_output_method',
              'embed_base_url', 'rerank_base_url', 'milvus_uri')
    config = {key: getattr(settings, key, None) for key in fields}
    root = Path(__file__).resolve().parents[2]
    config['pipeline_code'] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ('app/core/query_understanding.py', 'app/core/prompts.py',
                     'app/core/retrieval.py', 'app/core/rerank.py', 'app/core/embeddings.py')
    }
    return fingerprint(config)


def calibrate(rows: list[dict]) -> dict:
    positive, negative = [], []
    ignored = 0
    for row in rows:
        bucket = row['bucket']
        if bucket not in ANSWERABLE and bucket != 'D_absent':
            ignored += 1
            continue
        score = _number(row['score'])
        if not 0 <= score <= 1:
            raise ValueError('confidence must be between zero and one')
        (positive if bucket in ANSWERABLE else negative).append(score)
    report = {'status': 'pending_calibration', 'in_use': False,
              'counts': {'answerable': len(positive), 'absent': len(negative), 'ignored': ignored},
              'scan': [], 'recommended': None}
    if not positive or not negative:
        report['reason'] = 'both answerable and D_absent buckets are required'
        return report
    best_j = float('-inf')
    for step in range(5, 96):
        threshold = step / 100
        tpr = sum(score >= threshold for score in positive) / len(positive)
        fpr = sum(score >= threshold for score in negative) / len(negative)
        point = {'threshold': threshold, 'pass_rate': tpr, 'leak_rate': fpr, 'youden_j': tpr - fpr}
        report['scan'].append(point)
        if point['youden_j'] > best_j:
            best_j = point['youden_j']
            report['recommended'] = point.copy()
    report['status'] = 'ready'
    return report


def load_calibration(path: Path, expected: dict) -> dict:
    pending = {'status': 'pending_calibration', 'threshold': None, 'in_use': False}
    try:
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        meta = report['meta']
        if not META_KEYS <= expected.keys() or any(not expected[key] or meta.get(key) != expected[key] for key in META_KEYS):
            return pending
        if meta['signal_version'] != SIGNAL_VERSION or meta['strategy'] != 'hybrid_rerank':
            return pending
        threshold = _number(report['recommended']['threshold'])
        if report['status'] != 'ready' or not .05 <= threshold <= .95:
            return pending
        if report['counts']['answerable'] <= 0 or report['counts']['absent'] <= 0:
            return pending
        return {'status': 'ready', 'threshold': threshold, 'meta': meta, 'in_use': True}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return pending
