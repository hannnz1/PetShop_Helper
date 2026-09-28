from datetime import date
from types import SimpleNamespace

import pytest

from app.observability import dashboard
from app.observability.read_notes import build_read_note, matching_read_note


def rows():
    return [dict(day='2026-09-28', intent='knowledge', model_name='model', calls=3,
                 available_calls=2, unavailable_calls=1, input_tokens=200, output_tokens=100)]


def test_missing_usage_is_not_zero():
    summary = dashboard.summarize_usage(rows())
    assert summary['avg_tokens'] == 150
    assert summary['unavailable_calls'] == 1
    assert summary['rows'][0]['token_share'] == 1
    empty = dashboard.summarize_usage([dict(calls=1, available_calls=0, unavailable_calls=1,
                                           input_tokens=None, output_tokens=None)])
    assert empty['measured_tokens'] is None
    assert empty['avg_tokens'] is None and empty['token_share'] is None


def test_zero_tokens_have_no_share_and_note_is_bound():
    report = dashboard.summarize_usage([dict(calls=1, available_calls=1, unavailable_calls=0,
                                             input_tokens=0, output_tokens=0)])
    assert report['avg_tokens'] == 0
    assert report['token_share'] is None
    note = build_read_note('usage', report)
    assert matching_read_note('usage', report, note) == note['text']
    assert matching_read_note('usage', {**report, 'calls': 2}, note) is None


@pytest.mark.asyncio
async def test_overview_blocks_fail_independently(monkeypatch):
    async def usage(*args):
        return rows()

    async def trend(*args):
        raise ConnectionError('private db address must not leak')

    monkeypatch.setattr(dashboard.repository, 'usage_by_day', usage)
    monkeypatch.setattr(dashboard.repository, 'comparable_trend', trend)
    monkeypatch.setattr(dashboard, 'active_calibration', lambda s: {'status': 'pending_calibration', 'threshold': None})
    result = await dashboard.build_overview(date(2026, 9, 28), date(2026, 9, 28), 'a'*64, 'hybrid_rerank', settings=SimpleNamespace())
    assert result['cost']['data']['avg_tokens'] == 150
    assert result['trend'] == {'status': 'unavailable', 'error': 'ConnectionError'}
    assert result['calibration']['data']['status'] == 'pending_calibration'


@pytest.mark.asyncio
async def test_missing_default_dataset_does_not_hide_usage(tmp_path, monkeypatch):
    async def usage(*args):
        return rows()
    monkeypatch.setattr(dashboard.repository, 'usage_by_day', usage)
    monkeypatch.setattr(dashboard, 'CALIBRATION_DATASET', tmp_path / 'missing.jsonl')
    monkeypatch.setattr(dashboard, 'active_calibration', lambda s: {'status': 'pending_calibration', 'threshold': None})
    report = await dashboard.build_overview(date.today(), date.today(), None, 'hybrid_rerank', settings=SimpleNamespace())
    assert report['cost']['data']['avg_tokens'] == 150
    assert report['trend']['status'] == 'unavailable'
