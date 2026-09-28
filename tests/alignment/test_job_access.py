from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.jobs import router
from app.core import jobs


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(jobs.JOB_SPECS, 'private-test', SimpleNamespace(
        name='private-test', target='private-test', permission='observability', heavy=True))
    app = FastAPI()
    calls = []
    class Runner:
        def start(self, name, **kwargs):
            calls.append(name)
            return {'name': name}
        def stop(self, name):
            calls.append(name)
            return {'name': name}
        def status(self, name):
            calls.append(name)
            return {'name': name, 'log_tail': 'private log'}
        def list(self):
            return [{'name': 'private-test'}, {'name': 'kb-preview'}]
    app.state.jobs = Runner()
    app.state.settings = SimpleNamespace(observability_admin_token=SecretStr('correct'), knowledge_review_token=None)
    app.include_router(router)
    return TestClient(app), calls, app


@pytest.mark.parametrize('method,path', [('GET', ''), ('POST', ''), ('POST', '/stop')])
@pytest.mark.parametrize('authorization,status', [(None, 401), ('Bearer wrong', 401), ('Basic correct', 401), ('Bearer correct', 200)])
def test_sensitive_job_rejects_legacy_route_bypass(client, method, path, authorization, status):
    http, calls, _ = client
    headers = {'Authorization': authorization} if authorization else {}
    response = http.request(method, '/api/jobs/private-test' + path, headers=headers)
    assert response.status_code == status
    assert len(calls) == (1 if status == 200 else 0)


def test_unconfigured_and_unknown_jobs_fail_before_runner(client):
    http, calls, app = client
    app.state.settings.observability_admin_token = None
    assert http.post('/api/jobs/private-test').status_code == 503
    assert http.post('/api/jobs/missing').status_code == 404
    assert calls == []


def test_list_omits_private_jobs_without_access(client):
    http, _, _ = client
    assert http.get('/api/jobs').json()['jobs'] == [{'name': 'kb-preview'}]
    visible = http.get('/api/jobs', headers={'Authorization': 'Bearer correct'}).json()['jobs']
    assert {job['name'] for job in visible} == {'private-test', 'kb-preview'}


def test_classify_pool_job_requires_review_permission(client):
    http, calls, app = client
    assert jobs.JOB_SPECS['classify-pool'].permission == 'review'
    assert http.post('/api/jobs/classify-pool').status_code == 503
    app.state.settings.knowledge_review_token = SecretStr('review')
    assert http.post('/api/jobs/classify-pool').status_code == 401
    assert http.post('/api/jobs/classify-pool', headers={'Authorization': 'Bearer correct'}).status_code == 401
    assert calls == []
