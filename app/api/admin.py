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
    if not mysql["available"] and not milvus["available"]:
        status, summary = "unavailable", "MySQL 与 Milvus 读数失败"
    elif not mysql["available"]:
        status, summary = "unavailable", "MySQL 读数失败"
    elif not milvus["available"]:
        status, summary = "unavailable", "Milvus 读数失败"
    elif not stats["total"]:
        status, summary = "empty", "还没有知识块"
    elif stats["pending"] or knowledge["consistent"] is False:
        status, summary = "action", "有待向量化或双写不一致的知识块"
    else:
        status, summary = "normal", "知识库已同步"
    return {"modules": [
        {"id": "chat", "name": "纯对话客服", "href": "/", "status": "unknown",
         "summary": "入口可用；上游模型状态未在此页检测"},
        {"id": "agent", "name": "工具客服", "href": "/", "status": "unknown",
         "summary": "入口可用；业务工具状态未在此页检测"},
        {"id": "kb", "name": "知识库", "href": "/kb", "status": status,
         "summary": summary, "mysql": stats, "milvus_count": milvus["count"]},
    ]}
