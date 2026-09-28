from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import SecretStr


def test_operations_html_does_not_unlock_private_api(monkeypatch):
    from app.main import create_app
    from app.api import flywheel
    settings = SimpleNamespace(knowledge_review_token=SecretStr('review'), observability_admin_token=SecretStr('observe'))
    monkeypatch.setattr(flywheel, 'get_settings', lambda: settings)
    app = create_app()
    app.state.settings = settings
    client = TestClient(app)  # Static/auth routes do not need the model lifespan.
    for page in ('review', 'observability'):
        response = client.get('/' + page)
        assert response.status_code == 200
        assert 'text/html' in response.headers['content-type']
    assert client.get('/api/review/questions').status_code == 401
    assert client.get('/api/observability/overview').status_code == 401
    settings.knowledge_review_token = settings.observability_admin_token = None
    assert client.get('/api/review/questions').status_code == 503
    assert client.get('/api/observability/overview').status_code == 503
