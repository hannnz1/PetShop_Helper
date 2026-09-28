from fastapi import APIRouter, Depends
from app.api.observability import require_admin
from app.topics.acceptance import acceptance_overview, artifact_item

router = APIRouter(prefix='/api/acceptance', dependencies=[Depends(require_admin)])

@router.get('/overview')
async def overview():
    return await acceptance_overview()

@router.get('/data')
async def data():
    return {'items': [artifact_item(key) for key in ('golden', 'corpus', 'dataset')]}

@router.get('/evaluation')
async def evaluation():
    return artifact_item('evaluation')

@router.get('/errors')
async def errors():
    report = artifact_item('evaluation')
    body = report.get('data') or {}
    return {'status': report['status'], 'reason': report['reason'],
            'metadata': body.get('metadata', {}), 'errors': body.get('errors', [])}
