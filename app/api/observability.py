"""Token-only observability report with its own admin bearer credential."""

from datetime import date
from hmac import compare_digest
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.db import observability


router = APIRouter(prefix="/api/observability", tags=["observability"])
bearer = HTTPBearer(auto_error=False)


def require_admin(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    settings = getattr(request.app.state, "settings", None) or get_settings()
    token = settings.observability_admin_token
    if token is None or not token.get_secret_value().strip():
        raise HTTPException(status_code=503, detail="Observability report is not configured")
    if credentials is None or not compare_digest(
        credentials.credentials.encode("utf-8"), token.get_secret_value().encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid bearer token",
                            headers={"WWW-Authenticate": "Bearer"})


@router.get("/usage", dependencies=[Depends(require_admin)])
async def usage(
    start: Annotated[date, Query(alias="from")],
    end: Annotated[date, Query(alias="to")],
) -> list[dict]:
    if start > end:
        raise HTTPException(status_code=422, detail="from must be on or before to")
    return await observability.usage_by_day(start, end)


@router.get("/eval-trend", dependencies=[Depends(require_admin)])
async def eval_trend(
    dataset_hash: Annotated[str, Query(pattern=r"^[0-9a-f]{64}$")],
    strategy: Annotated[str, Query(pattern=r"^(vector|bm25|hybrid|hybrid_rerank)$")],
) -> list[dict]:
    return await observability.comparable_trend(dataset_hash, strategy)
