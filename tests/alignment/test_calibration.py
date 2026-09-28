import json
from types import SimpleNamespace

import pytest

from app.core.evidence_confidence import calibrate, load_calibration, SIGNAL_VERSION


def test_lowest_threshold_wins_tie():
    result = calibrate([{'bucket': 'A_policy', 'score': .8}, {'bucket': 'D_absent', 'score': .2}])
    assert result['recommended']['threshold'] == .21
    assert len(result['scan']) == 91


def test_missing_negative_bucket_is_pending():
    assert calibrate([{'bucket': 'A_policy', 'score': .8}])['status'] == 'pending_calibration'
    assert calibrate([{'bucket': 'D_absent', 'score': .2}])['status'] == 'pending_calibration'


def test_report_requires_matching_provenance(tmp_path):
    path = tmp_path / 'calibration.json'
    expected = dict(signal_version=SIGNAL_VERSION, dataset_hash='dataset', strategy='hybrid_rerank', retrieval_config_hash='config')
    assert load_calibration(path, expected)['status'] == 'pending_calibration'
    report = calibrate([{'bucket': 'A_policy', 'score': .8}, {'bucket': 'D_absent', 'score': .2}])
    report['meta'] = expected
    path.write_text(json.dumps(report), encoding='utf-8')
    assert load_calibration(path, expected)['threshold'] == .21
    assert load_calibration(path, {**expected, 'retrieval_config_hash': 'changed'})['status'] == 'pending_calibration'
    path.write_text('{broken', encoding='utf-8')
    assert load_calibration(path, expected)['status'] == 'pending_calibration'


def test_offline_cli_is_repeatable_and_does_not_retrieve(tmp_path, monkeypatch):
    from scripts.ch09 import calibrate_confidence as cli

    async def forbidden(*args):
        raise AssertionError('offline CLI must not use retrieval')

    monkeypatch.setattr(cli, 'live_scores', forbidden)
    source = tmp_path / 'scores.jsonl'
    source.write_text('\n'.join(json.dumps(row) for row in [
        {'bucket': 'A_policy', 'score': .8}, {'bucket': 'D_absent', 'score': .2},
    ]), encoding='utf-8')
    first, second = tmp_path / 'one.json', tmp_path / 'two.json'
    for output in (first, second):
        assert cli.main(['--scores-file', str(source), '--out', str(output)]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text())['in_use'] is False


def test_offline_bound_report_loads_in_runtime(tmp_path, monkeypatch):
    from app.core import faq_pipeline
    from app.core.evidence_confidence import retrieval_fingerprint
    from scripts.ch09.calibrate_confidence import main

    dataset, scores, report = (tmp_path / name for name in ('dataset.jsonl', 'scores.jsonl', 'report.json'))
    samples = [{'id': 'A1', 'bucket': 'A_policy'}, {'id': 'D1', 'bucket': 'D_absent'}]
    dataset.write_text('\n'.join(json.dumps(row) for row in samples), encoding='utf-8')
    scores.write_text('\n'.join(json.dumps({**row, 'score': score}) for row, score in zip(samples, [.8, .2])), encoding='utf-8')
    settings = SimpleNamespace(embed_model='embed', rerank_model='rerank', recall_top_k=50,
                               rerank_top_k=10, evidence_calibration_path=str(report))
    monkeypatch.setattr(faq_pipeline, 'CALIBRATION_DATASET', dataset)
    main(['--scores-file', str(scores), '--dataset', str(dataset), '--retrieval-config-hash',
          retrieval_fingerprint(settings), '--out', str(report)])
    assert faq_pipeline.active_calibration(settings)['threshold'] == .21


@pytest.mark.asyncio
async def test_live_calibration_uses_runtime_rewrite(monkeypatch):
    from app.core import faq_pipeline
    from scripts.ch09.calibrate_confidence import live_scores

    async def understand(query):
        assert query == '原问题'
        return {'standard': '规范问题', 'expanded': ['同义词']}

    async def search(query, **kwargs):
        assert query == '规范问题'
        assert kwargs == {'strategy': 'hybrid_rerank', 'category': None, 'bm25_query': '规范问题 同义词'}
        return [{'rerank_score': .8}]

    monkeypatch.setattr(faq_pipeline.query_understanding, 'understand', understand)
    monkeypatch.setattr(faq_pipeline.retrieval, 'search_knowledge', search)
    rows = await live_scores([{'id': 'A1', 'query': '原问题', 'bucket': 'A_policy'}])
    assert rows[0]['score'] > 0
