"""Small module cards for the local development administration home."""

from fastapi import APIRouter, Request

from app.api import kb


router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/overview")
async def overview(request: Request) -> dict:
    knowledge = await kb.overview(request)
    mysql = knowledge["mysql"]
    milvus = knowledge["milvus"]
    stats = mysql["stats"] if mysql["available"] else None
    if not mysql["available"] or not milvus["available"]:
        status, summary = "unavailable", "知识库依赖读数失败"
    elif not stats["total"]:
        status, summary = "empty", "还没有知识块"
    elif stats["pending"] or knowledge["consistent"] is False:
        status, summary = "action", "有待向量化或双写不一致的知识块"
    else:
        status, summary = "normal", "知识库已同步"
    return {"modules": [
        {"id": "chat", "name": "纯对话客服", "href": "/", "status": "normal",
         "summary": "多轮对话与结构化售后提取"},
        {"id": "agent", "name": "工具客服", "href": "/", "status": "normal",
         "summary": "工具调用与业务处理"},
        {"id": "kb", "name": "知识库", "href": "/kb", "status": status,
         "summary": summary, "mysql": stats, "milvus_count": milvus["count"]},
    ]}
