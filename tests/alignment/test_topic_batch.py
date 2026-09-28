import asyncio
import pytest
from sqlalchemy import select, func
from app.core.taxonomy import TOPIC_NAMES
from app.db.models import LowConfidenceQuestion
from tests.alignment.test_classifier_metadata import metadata


async def seed(factory, count):
    async with factory.begin() as session:
        session.add_all([LowConfidenceQuestion(raw_question=f'问题{i} 联系13800138000',
                         source='user_feedback') for i in range(count)])


@pytest.mark.asyncio
async def test_batch_is_persisted_once(db_session_factory, db_clean, monkeypatch):
    from app.topics import batch
    from app.db.models import TopicClassification
    await seed(db_session_factory, 10)
    async def health(): return metadata()
    async def classify(texts, expected):
        assert all('13800138000' not in text for text in texts)
        return {**metadata(), 'results': [{'labels': [TOPIC_NAMES[0]],
                'scores': dict.fromkeys(TOPIC_NAMES, .6)} for _ in texts]}
    monkeypatch.setattr(batch, 'health_metadata', health)
    monkeypatch.setattr(batch, 'classify_batch', classify)
    first = await batch.classify_pool()
    second = await batch.classify_pool()
    assert first['written'] == 10 and first['status'] == 'done'
    assert second['status'] == 'empty'
    async with db_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(TopicClassification)) == 10


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['short', 'extra', 'labels', 'version', 'network', 'ok'])
async def test_batches_validate_all_before_writing(db_session_factory, db_clean, monkeypatch, mode):
    from app.topics import batch
    from app.db.models import TopicClassification
    await seed(db_session_factory, 101)
    calls = []
    async def health(): return metadata()
    async def classify(texts, expected):
        calls.append(len(texts))
        result = {**metadata(), 'results': [{'labels': [TOPIC_NAMES[0]],
                 'scores': dict.fromkeys(TOPIC_NAMES, .6)} for _ in texts]}
        if len(calls) == 2:
            if mode == 'short': result['results'] = []
            if mode == 'extra': result['results'] *= 2
            if mode == 'labels': result['results'][0]['labels'] = ['非法']
            if mode == 'version': result['model_version'] = 'b'*64
            if mode == 'network': raise TimeoutError()
        return result
    monkeypatch.setattr(batch, 'health_metadata', health)
    monkeypatch.setattr(batch, 'classify_batch', classify)
    report = await batch.classify_pool()
    assert calls == [100, 1]
    async with db_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(TopicClassification)) == (101 if mode == 'ok' else 0)
    assert report['status'] == ('done' if mode == 'ok' else 'failed')


@pytest.mark.asyncio
async def test_below_batch_and_concurrent_writes(db_session_factory, db_clean, monkeypatch):
    from app.topics import batch
    from app.db import topics
    await seed(db_session_factory, 9)
    async def forbidden(): raise AssertionError('must not call service')
    monkeypatch.setattr(batch, 'health_metadata', forbidden)
    assert (await batch.classify_pool())['status'] == 'below_batch'
    rows = await topics.list_unclassified(500)
    rows = [{**row, 'labels': [TOPIC_NAMES[0]]} for row in rows]
    results = await asyncio.gather(*(topics.write_classifications(rows, metadata(), run_id)
                                    for run_id in ('parallel-1', 'parallel-2')))
    assert sum(row['written'] for row in results) == 9
    assert sum(row['question_count'] for row in results) == 9
    assert await topics.list_unclassified(500) == []


@pytest.mark.asyncio
async def test_audit_failure_rolls_back_classifications(db_session_factory, db_clean, monkeypatch):
    from app.db import topics
    from app.db.models import TopicClassification
    await seed(db_session_factory, 10)
    rows = [{**row, 'labels': [TOPIC_NAMES[0]]} for row in await topics.list_unclassified(10)]
    def broken_audit(**kwargs): raise RuntimeError('audit write failure')
    monkeypatch.setattr(topics, 'TopicClassificationRun', broken_audit)
    with pytest.raises(RuntimeError, match='audit'):
        await topics.write_classifications(rows, metadata(), 'audit-failure')
    async with db_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(TopicClassification)) == 0


@pytest.mark.asyncio
async def test_force_and_explicit_reclassification(db_session_factory, db_clean, monkeypatch):
    from app.topics import batch
    from app.db.models import TopicClassification
    await seed(db_session_factory, 9)
    meta = metadata()
    async def health(): return dict(meta)
    async def classify(texts, expected):
        return {**meta, 'results': [{'labels': [TOPIC_NAMES[0]],
                'scores': dict.fromkeys(TOPIC_NAMES, .6)} for _ in texts]}
    monkeypatch.setattr(batch, 'health_metadata', health)
    monkeypatch.setattr(batch, 'classify_batch', classify)
    assert (await batch.classify_pool(force=True))['written'] == 9
    old = meta['model_version']
    meta['model_version'] = 'b'*64
    assert (await batch.classify_pool(force=True))['status'] == 'empty'
    result = await batch.classify_pool(force=True, reclassify=True)
    assert result['previous_versions'] == [old] and result['written'] == 9
    async with db_session_factory() as session:
        assert set((await session.scalars(select(TopicClassification.model_version))).all()) == {'b'*64}
