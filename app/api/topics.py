from fastapi import APIRouter, Depends, HTTPException, Query
from app.api.flywheel import require_review_token
from app.core.taxonomy import TOPIC_NAMES
from app.db.topics import topic_distribution, topic_questions

router = APIRouter(prefix='/api/topics', dependencies=[Depends(require_review_token)])

@router.get('/distribution')
async def distribution():
    return await topic_distribution()

@router.get('/questions')
async def questions(label: str, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
    if label not in TOPIC_NAMES:
        raise HTTPException(422, 'Unknown topic label')
    return await topic_questions(label, offset, limit)
