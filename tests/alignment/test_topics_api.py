import pytest
import httpx
from fastapi import FastAPI
from pydantic import SecretStr
from app.config import get_settings
from app.core.taxonomy import TOPIC_NAMES
from tests.alignment.test_topic_batch import seed
from tests.alignment.test_classifier_metadata import metadata


@pytest.mark.asyncio
async def test_multilabel_distribution_keeps_unique_question_count(db_session_factory, db_clean):
    from app.db import topics
    await seed(db_session_factory, 3)
    rows = await topics.list_unclassified(10)
    await topics.write_classifications([{**rows[0], 'labels': list(TOPIC_NAMES[:2])},
                                       {**rows[1], 'labels': [TOPIC_NAMES[0]]}], metadata(), 'topics-fixture')
    report = await topics.topic_distribution()
    assert report['question_count'] == 2 and report['unclassified_count'] == 1
    assert sum(report['class_counts'].values()) == 3 and len(report['class_counts']) == 17
    first = await topics.topic_questions(TOPIC_NAMES[0], 0, 1)
    second = await topics.topic_questions(TOPIC_NAMES[0], 1, 1)
    assert first['total'] == second['total'] == 2
    assert first['items'][0]['id'] != second['items'][0]['id']
    assert '13800138000' not in str(first)
    assert first['items'][0]['review'] is None


@pytest.mark.asyncio
async def test_topics_access_and_labels(monkeypatch, db_session_factory, db_clean):
    from app.api.topics import router
    from app.api import flywheel
    settings = get_settings().model_copy(update={'knowledge_review_token': SecretStr('topic-test')})
    monkeypatch.setattr(flywheel, 'get_settings', lambda: settings)
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
        assert (await client.get('/api/topics/distribution')).status_code == 401
        headers = {'Authorization': 'Bearer topic-test'}
        assert (await client.get('/api/topics/distribution', headers=headers)).json()['question_count'] == 0
        assert (await client.get('/api/topics/questions', params={'label':'bad'}, headers=headers)).status_code == 422
        settings.knowledge_review_token = None
        assert (await client.get('/api/topics/distribution', headers=headers)).status_code == 503
