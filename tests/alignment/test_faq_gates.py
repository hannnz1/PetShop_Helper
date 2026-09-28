from types import SimpleNamespace

import pytest

from app.core import faq_pipeline as pipeline


@pytest.fixture
def setup(monkeypatch):
    settings = SimpleNamespace(milvus_uri='http://127.0.0.1:19530', rerank_min_score=.3)

    async def understand(query):
        return {'standard': query, 'expanded': ['扩展']}

    async def check(query, evidence):
        return {'useful': True, 'reason': ''}

    monkeypatch.setattr(pipeline.query_understanding, 'understand', understand)
    monkeypatch.setattr(pipeline.selfcheck, 'check_sufficient', check)
    return settings


def hit(score=.8):
    return {'id': 1, 'rerank_score': score, 'question': '型号', 'answer': '参数',
            'section_path': '手册', 'content_type': 'manual'}


@pytest.mark.asyncio
async def test_wrong_category_retries_once(setup, monkeypatch):
    categories = []

    async def search(query, **kwargs):
        categories.append(kwargs['category'])
        assert query == 'MH-W40'
        assert kwargs['bm25_query'] == 'MH-W40 扩展'
        return [] if kwargs['category'] else [hit()]

    monkeypatch.setattr(pipeline.retrieval, 'search_knowledge', search)
    result = await pipeline.run_faq_pipeline('MH-W40', '猜错的分类', settings=setup)
    assert result['sufficient'] is True
    assert categories == ['猜错的分类', None]


@pytest.mark.asyncio
async def test_confidence_rejection_precedes_selfcheck(setup, monkeypatch):
    async def search(*args, **kwargs):
        return [hit(.4)]

    async def forbidden(*args):
        raise AssertionError('selfcheck must not run for rejected evidence')

    monkeypatch.setattr(pipeline.retrieval, 'search_knowledge', search)
    monkeypatch.setattr(pipeline.selfcheck, 'check_sufficient', forbidden)
    monkeypatch.setattr(pipeline, 'active_calibration', lambda settings: {'status': 'ready', 'threshold': .9})
    result = await pipeline.run_faq_pipeline('型号', gate='calibrated', settings=setup)
    assert result['sufficient'] is False
    assert result['trace']['calibration']['threshold'] == .9


@pytest.mark.asyncio
async def test_missing_calibration_retains_top1(setup, monkeypatch):
    async def search(*args, **kwargs):
        return [hit(.4)]

    monkeypatch.setattr(pipeline.retrieval, 'search_knowledge', search)
    monkeypatch.setattr(pipeline, 'active_calibration', lambda settings: {'status': 'pending_calibration', 'threshold': None})
    result = await pipeline.run_faq_pipeline('型号', gate='calibrated', settings=setup)
    assert result['sufficient'] is True
    assert result['trace']['calibration']['status'] == 'pending_calibration'


@pytest.mark.asyncio
async def test_category_retry_never_repeats_selfcheck(setup, monkeypatch):
    calls = []

    async def search(*args, **kwargs):
        return [hit()]

    async def check(*args):
        calls.append(1)
        return {'useful': False, 'reason': '不支持'}

    monkeypatch.setattr(pipeline.retrieval, 'search_knowledge', search)
    monkeypatch.setattr(pipeline.selfcheck, 'check_sufficient', check)
    assert not (await pipeline.run_faq_pipeline('型号', '商品', settings=setup))['sufficient']
    assert calls == [1]
