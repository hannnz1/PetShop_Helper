"""Chapter-three routers are mounted on the actual application factory."""

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_knowledge_and_jobs_routes_are_reachable_in_application():
    settings = Settings(
        _env_file=None, chat_model="test", chat_base_url="https://example.test/v1",
        chat_api_key="test",
    )
    with TestClient(create_app(settings=settings, model=object())) as client:
        preview = client.post("/api/kb/preview", json={
            "content_type": "faq", "markdown": "# 运费\n\n满99元包邮。",
        })
        jobs = client.get("/api/jobs")
        admin = client.get("/admin")
        kb_page = client.get("/kb")
        nav = client.get("/static/admin.js")
        admin_api = client.get("/api/admin/overview")
    assert preview.status_code == 200
    assert jobs.status_code == 200
    assert jobs.json()["jobs"]
    assert admin.status_code == kb_page.status_code == nav.status_code == 200
    assert "/static/admin.js" in admin.text
    assert "/static/admin.js" in kb_page.text
    assert admin_api.status_code == 200
