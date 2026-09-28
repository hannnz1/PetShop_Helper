"""Whitelisted administrative Make job endpoints."""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Annotated
from pydantic import BaseModel
from app.config import get_settings
from app.core.admin_access import require_job_access

from app.core.jobs import (
    ConfirmationRequired,
    JobAlreadyRunning,
    JobRunner,
    UnknownJob,
)


router = APIRouter(prefix="/api/jobs", tags=["jobs"])
bearer = HTTPBearer(auto_error=False)
Credentials = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]


class StartJobRequest(BaseModel):
    confirm: bool = False


def _runner(request: Request) -> JobRunner:
    return request.app.state.jobs


def _access(name: str, request: Request, credentials) -> None:
    settings = getattr(request.app.state, 'settings', None) or get_settings()
    require_job_access(name, credentials.credentials if credentials else None, settings)


def _visible(name: str, request: Request, credentials) -> bool:
    try:
        _access(name, request, credentials)
        return True
    except HTTPException:
        return False


@router.get("")
def list_jobs(request: Request, credentials: Credentials) -> dict[str, object]:
    return {"jobs": [item for item in _runner(request).list() if _visible(item['name'], request, credentials)]}


@router.post("/{name}")
def start_job(name: str, request: Request, credentials: Credentials, body: StartJobRequest | None = None) -> dict[str, object]:
    _access(name, request, credentials)
    try:
        return _runner(request).start(name, confirm=body.confirm if body else False)
    except UnknownJob:
        raise HTTPException(status_code=404, detail="Unknown job") from None
    except ConfirmationRequired:
        raise HTTPException(status_code=400, detail="Confirmation required") from None
    except JobAlreadyRunning:
        raise HTTPException(status_code=409, detail="Job is already running") from None
    except (FileNotFoundError, OSError):
        raise HTTPException(status_code=503, detail="Job runner unavailable") from None


@router.get("/{name}")
def get_job(name: str, request: Request, credentials: Credentials) -> dict[str, object]:
    _access(name, request, credentials)
    try:
        return _runner(request).status(name)
    except UnknownJob:
        raise HTTPException(status_code=404, detail="Unknown job") from None


@router.post("/{name}/stop")
def stop_job(name: str, request: Request, credentials: Credentials) -> dict[str, object]:
    _access(name, request, credentials)
    try:
        return _runner(request).stop(name)
    except UnknownJob:
        raise HTTPException(status_code=404, detail="Unknown job") from None
    except (OSError, RuntimeError):
        raise HTTPException(status_code=503, detail="Unable to stop job") from None
