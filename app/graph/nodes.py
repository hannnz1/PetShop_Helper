"""Chapter 5 workflow nodes, beginning with the mandatory knowledge gate."""

from app.db import repository
from app.tools.business import query_faq


def _strong_evidence(payload: object) -> bool:
    if not isinstance(payload, dict) or payload.get("sufficient") is not True:
        return False
    citations = payload.get("citations")
    return (
        isinstance(payload.get("evidence"), str)
        and bool(payload["evidence"].strip())
        and isinstance(citations, list)
        and bool(citations)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("n"), int)
            and isinstance(item.get("answer"), str)
            and isinstance(item.get("section_path"), str)
            for item in citations
        )
    )


async def forced_rag(state: dict) -> dict:
    """Always retrieve first; malformed or failed evidence cannot enter Agent."""
    query = state["query"]
    try:
        payload = await query_faq.ainvoke({"keyword": query})
    except Exception as exc:  # noqa: BLE001 - retrieval failure is a refusal
        payload = {"sufficient": False, "source": "self_check",
                   "reason": f"知识检索失败: {type(exc).__name__}"}

    if _strong_evidence(payload):
        return {
            "sufficient": True, "evidence": payload["evidence"],
            "citations": payload["citations"], "reason": "",
        }

    source = payload.get("source") if isinstance(payload, dict) else None
    if source not in {"retrieval_low_conf", "self_check"}:
        source = "self_check"
    reason = payload.get("reason") if isinstance(payload, dict) else None
    if not isinstance(reason, str) or not reason.strip():
        reason = "知识库证据格式不完整或不足"
    await repository.insert_low_confidence(state["conversation_id"], query, source, reason)
    return {"sufficient": False, "evidence": "", "citations": [],
            "source": source, "reason": reason}


def confidence_gate(state: dict) -> str:
    return "agent" if _strong_evidence(state) else "fallback"
