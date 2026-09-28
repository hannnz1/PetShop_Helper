"""Small module cards for the local development administration home."""

from fastapi import APIRouter, Request

from app.api import kb, rageval


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
    elif stats["pending"] or knowledge["consistent"] is False:
        status, summary = "action", "有待向量化或双写不一致的知识块"
    elif not stats["total"]:
        status, summary = "empty", "还没有知识块"
    else:
        status, summary = "normal", "知识库已同步"
    report = rageval._report()
    best = rageval._best(report["retrieval"]) if report else None
    rag_metrics = {"题数": report["meta"].get("question_count", 0)} if report else {}
    if best:
        rag_metrics["最佳 MRR"] = round(best["mrr"], 3)
    if report and report.get("generation"):
        refusal = report["generation"].get("refusal_rate")
        if refusal is not None:
            rag_metrics["库外拒答率"] = f"{refusal:.0%}"
    return {"modules": [
        {"id": "topics", "name": "问题主题", "href": "/topics", "status": "unknown",
         "summary": "审核凭据访问17类统计、脱敏问题和本机分类批次"},
        {"id": "acceptance", "name": "分类器验收", "href": "/acceptance", "status": "unknown",
         "summary": "观测凭据查看九项报告与数据、模型版本"},
        {"id": "manual-test-samples", "name": "逐章演示", "href": "/manual-test-samples", "status": "unknown",
         "summary": "可复制问句、验证步骤及真实与模拟数据说明"},
        {"id": "review", "name": "人工审核", "href": "/review", "status": "unknown",
         "summary": "凭审核凭据查看来源、批准答案及发布状态"},
        {"id": "observability", "name": "观测与用量", "href": "/observability", "status": "unknown",
         "summary": "凭观测凭据查看真实用量、评估趋势和校准结果"},
        {"id": "chat", "name": "纯对话客服", "href": "/", "status": "unknown",
         "summary": "入口可用；上游模型状态未在此页检测"},
        {"id": "agent", "name": "工具客服", "href": "/", "status": "unknown",
         "summary": "入口可用；业务工具状态未在此页检测"},
        {"id": "kb", "name": "知识库", "href": "/kb", "status": status,
         "summary": summary, "mysql": stats, "milvus_count": milvus["count"]},
        {"id": "rag-eval", "name": "RAG 评估", "href": "/rag-eval",
         "status": "normal" if report else "empty",
         "summary": "四策略报告可查看" if report else "尚未生成评估报告",
         "metrics": rag_metrics},
    ]}
