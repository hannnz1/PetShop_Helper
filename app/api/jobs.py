"""Whitelisted administrative Make job endpoints."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.core.jobs import (
    ConfirmationRequired,
    JobAlreadyRunning,
    JobRunner,
    UnknownJob,
)


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class StartJobRequest(BaseModel):
    confirm: bool = False


def _runner(request: Request) -> JobRunner:
    return request.app.state.jobs


@router.get("")
def list_jobs(request: Request) -> dict[str, object]:
    return {"jobs": _runner(request).list()}


@router.post("/{name}")
def start_job(name: str, request: Request, body: StartJobRequest | None = None) -> dict[str, object]:
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
def get_job(name: str, request: Request) -> dict[str, object]:
    try:
        return _runner(request).status(name)
    except UnknownJob:
        raise HTTPException(status_code=404, detail="Unknown job") from None


@router.post("/{name}/stop")
def stop_job(name: str, request: Request) -> dict[str, object]:
    try:
        return _runner(request).stop(name)
    except UnknownJob:
        raise HTTPException(status_code=404, detail="Unknown job") from None
    except (OSError, RuntimeError):
        raise HTTPException(status_code=503, detail="Unable to stop job") from None
