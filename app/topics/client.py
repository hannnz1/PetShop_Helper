"""Strict local classifier boundary; no fallback labels on failures."""

import hashlib
import math
import re
import httpx

from app.config import get_settings
from app.core.taxonomy import TOPIC_NAMES, terminology_table


def validate_metadata(payload: dict, expected: dict | None = None) -> dict:
    if not isinstance(payload, dict):
        raise ValueError('classifier metadata must be an object')
    meta = {key: payload.get(key) for key in ('model_version', 'taxonomy_hash', 'threshold')}
    if not isinstance(meta['model_version'], str) or not re.fullmatch('[a-f0-9]{64}', meta['model_version']):
        raise ValueError('invalid classifier version')
    if meta['taxonomy_hash'] != hashlib.sha256(terminology_table().encode('utf-8')).hexdigest():
        raise ValueError('classifier taxonomy mismatch')
    threshold = meta['threshold']
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('invalid classifier threshold')
    if expected is not None and any(meta[key] != expected.get(key) for key in meta):
        raise ValueError('classifier version changed between batches')
    return meta


def validate_response(payload: dict, count: int, expected: dict) -> dict:
    validate_metadata(payload, expected)
    rows = payload.get('results')
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError('classifier result length mismatch')
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('invalid classifier result')
        labels, scores = row.get('labels'), row.get('scores')
        if (not isinstance(labels, list) or not labels or any(not isinstance(x, str) for x in labels)
                or len(labels) != len(set(labels)) or not set(labels) <= set(TOPIC_NAMES)):
            raise ValueError('invalid classifier labels')
        if not isinstance(scores, dict) or set(scores) != set(TOPIC_NAMES):
            raise ValueError('invalid classifier scores')
        if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in scores.values()):
            raise ValueError('invalid classifier score value')
    return payload


async def health_metadata() -> dict:
    settings = get_settings()
    async with httpx.AsyncClient(base_url=settings.classifier_base_url, timeout=settings.classifier_timeout,
                                 trust_env=False, follow_redirects=False) as client:
        response = await client.get('/healthz')
        response.raise_for_status()
        body = response.json()
        if body.get('ready') is not True:
            raise ValueError('classifier not ready')
        return validate_metadata(body)


async def classify_batch(texts: list[str], expected: dict) -> dict:
    if not 1 <= len(texts) <= 100 or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('classifier requires 1..100 nonblank texts')
    settings = get_settings()
    async with httpx.AsyncClient(base_url=settings.classifier_base_url, timeout=settings.classifier_timeout,
                                 trust_env=False, follow_redirects=False) as client:
        response = await client.post('/classify', json={'texts': texts})
        response.raise_for_status()
        return validate_response(response.json(), len(texts), expected)
