"""Read-only dashboard sections backed by measured MySQL usage."""

import asyncio
import hashlib
import json
from pathlib import Path

from app.config import get_settings
from app.core.faq_pipeline import active_calibration, CALIBRATION_DATASET
from app.db import observability as repository
from app.observability.read_notes import build_read_note


def summarize_usage(rows: list[dict]) -> dict:
    calls = sum(row['calls'] for row in rows)
    available = sum(row['available_calls'] for row in rows)
    unavailable = sum(row['unavailable_calls'] for row in rows)
    measured = sum((row['input_tokens'] or 0) + (row['output_tokens'] or 0)
                   for row in rows if row['available_calls']) if available else None
    result = {'calls': calls, 'available_calls': available, 'unavailable_calls': unavailable,
              'measured_tokens': measured, 'avg_tokens': measured / available if available else None,
              'token_share': 1.0 if measured else None}
    result['rows'] = []
    for row in rows:
        total = (row['input_tokens'] or 0) + (row['output_tokens'] or 0) if row['available_calls'] else None
        result['rows'].append({**row, 'measured_tokens': total,
                               'avg_tokens': total / row['available_calls'] if row['available_calls'] else None,
                               'token_share': total / measured if total is not None and measured else None})
    return result


async def build_overview(start, end, dataset_hash, strategy, *, settings=None) -> dict:
    if start > end:
        raise ValueError('start must be on or before end')
    settings = settings or get_settings()

    async def cost():
        return summarize_usage(await repository.usage_by_day(start, end))

    async def trend():
        current_hash = dataset_hash or hashlib.sha256(CALIBRATION_DATASET.read_bytes()).hexdigest()
        return {'dataset_hash': current_hash, 'rows': await repository.comparable_trend(current_hash, strategy)}

    async def calibration():
        active = active_calibration(settings)
        if active['status'] == 'ready':
            report = json.loads(Path(settings.evidence_calibration_path).read_text(encoding='utf-8'))
            return {**active, 'counts': report['counts'], 'scan': report['scan'], 'recommended': report['recommended']}
        return active

    async def section(kind, operation):
        try:
            data = await operation()
            return {'status': 'available', 'data': data, 'note': build_read_note(kind, data)}
        except Exception as exc:
            return {'status': 'unavailable', 'error': type(exc).__name__}

    results = await asyncio.gather(section('usage', cost), section('trend', trend), section('calibration', calibration))
    return dict(zip(('cost', 'trend', 'calibration'), results))
