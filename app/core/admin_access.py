"""One permission check shared by every administrative job endpoint."""

from hmac import compare_digest

from fastapi import HTTPException

from app.core.jobs import JOB_SPECS


def require_job_access(name: str, credentials: str | None, settings) -> None:
    spec = JOB_SPECS.get(name)
    if spec is None:
        raise HTTPException(status_code=404, detail='Unknown job')
    if spec.permission is None:
        return
    field = {'review': 'knowledge_review_token', 'observability': 'observability_admin_token'}.get(spec.permission)
    configured = getattr(settings, field, None) if field else None
    if configured is None or not configured.get_secret_value().strip():
        raise HTTPException(status_code=503, detail='Job credential not configured')
    if credentials is None or not compare_digest(credentials.encode(), configured.get_secret_value().encode()):
        raise HTTPException(status_code=401, detail='Invalid bearer token', headers={'WWW-Authenticate': 'Bearer'})
