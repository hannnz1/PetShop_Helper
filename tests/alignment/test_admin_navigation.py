import pytest
import httpx


@pytest.mark.asyncio
async def test_every_navigation_target_is_a_real_html_route():
    from app.main import create_app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app()), base_url='http://test') as client:
        for path in ('/', '/admin', '/kb', '/rag-eval', '/review', '/observability', '/topics',
                     '/topics/questions', '/acceptance', '/acceptance/data', '/acceptance/evaluation',
                     '/acceptance/errors', '/manual-test-samples'):
            response = await client.get(path)
            assert response.status_code == 200, path
            assert response.headers['content-type'].startswith('text/html')
        assert (await client.get('/not-a-page')).status_code == 404
