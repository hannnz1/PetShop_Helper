"""Knowledge-base review and ingestion endpoints."""

from dataclasses import asdict, replace

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.core import retrieval
from app.db import repository
from app.kb import documents, dualwrite, milvus_client, sources
from scripts.build_kb import source_chunks


router = APIRouter(prefix="/api/kb", tags=["knowledge-base"])


def _source_inventory() -> list[dict]:
    inventory = []
    for filename, content_type in sources.SOURCE_TYPES.items():
        path, _ = sources.resolve_source(filename)
        try:
            chunks = source_chunks(filename, path.read_text(encoding="utf-8"), content_type)
            inventory.append({
                "filename": filename, "content_type": content_type,
                "count": len(chunks),
                "key_clauses": sum(chunk.is_key_clause for chunk in chunks),
            })
        except Exception:
            inventory.append({
                "filename": filename, "content_type": content_type,
                "count": None, "key_clauses": None,
            })
    return inventory


@router.get("/overview")
async def overview(request: Request) -> dict:
    """Read each inventory source independently so outages stay local."""
    mysql = {"available": False, "stats": None, "recent": []}
    staging = None
    milvus = {"available": False, "count": None}
    try:
        stats = await repository.knowledge_stats()
        mysql = {"available": True, "stats": stats, "recent": []}
        try:
            rows = await repository.list_recent_chunks()
            mysql["recent"] = [{
                "id": row.id, "category": row.category,
                "questions": row.questions, "answer": row.answer,
                "content_type": row.content_type,
                "vectorize_status": row.vectorize_status,
            } for row in rows]
        except Exception:
            pass
    except Exception:
        pass
    try:
        staging = await repository.staging_stats()
    except Exception:
        pass
    try:
        client = milvus_client.get_client()
        try:
            milvus = {
                "available": True,
                "count": milvus_client.count(client)
                if client.has_collection(milvus_client.COLLECTION) else 0,
            }
        finally:
            client.close()
    except Exception:
        pass
    consistent = None
    if mysql["available"] and milvus["available"]:
        consistent = mysql["stats"]["done"] == milvus["count"]
    runner = getattr(request.app.state, "jobs", None)
    return {
        "mysql": mysql, "milvus": milvus, "consistent": consistent,
        "staging": staging, "sources": _source_inventory(),
        "jobs": runner.list() if runner is not None else [],
    }


@router.get("/staging")
async def staging_rows() -> dict:
    try:
        rows = []
        for status in ("extracted", "kept", "discarded"):
            rows.extend(await repository.list_staging_by_status(status))
    except Exception as exc:
        raise HTTPException(503, "staging unavailable") from exc
    rows.sort(key=lambda row: row.id, reverse=True)
    return {"rows": [{
        "id": row.id, "batch_no": row.batch_no, "source_ref": row.source_ref,
        "question": row.question, "answer": row.answer, "status": row.status,
    } for row in rows]}


class SourceRequest(BaseModel):
    content_type: str | None = None
    markdown: str | None = None
    filename: str | None = None
    vectorize: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = Field(default=None, ge=1, le=50)
    min_score: float | None = Field(default=None, ge=-1, le=1)


def _chunks(request: SourceRequest) -> list[documents.Chunk]:
    if request.filename is not None:
        if request.markdown is not None or request.content_type is not None:
            raise HTTPException(400, "filename cannot be combined with markdown/content_type")
        try:
            path, content_type = sources.resolve_source(request.filename)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        markdown = path.read_text(encoding="utf-8")
        chunks = source_chunks(request.filename, markdown, content_type)
    else:
        if request.content_type not in {"faq", "policy", "manual"}:
            raise HTTPException(400, "invalid content_type")
        markdown = (request.markdown or "").strip()
        if not markdown:
            raise HTTPException(400, "markdown is required")
        chunks = [
            replace(chunk, section_path=f"manual :: {chunk.section_path}")
            for chunk in documents.build_chunks(markdown, request.content_type)
        ]
    if not chunks:
        raise HTTPException(400, "source contains no knowledge chunks")
    return chunks


@router.post("/preview")
async def preview(request: SourceRequest) -> dict:
    """Show the exact chunks that ingestion would write, without side effects."""
    chunks = _chunks(request)
    try:
        existing = {
            repository.manual_pair_key(question, answer)
            for question, answer in await repository.list_chunk_pairs()
        } if request.filename is None else None
    except Exception:
        existing = None
    shown = []
    for chunk in chunks:
        row = asdict(chunk)
        row["duplicate"] = None if existing is None else (
            repository.manual_pair_key(chunk.questions, chunk.answer) in existing
        )
        shown.append(row)
    return {"count": len(chunks), "chunks": shown}


@router.post("/ingest")
async def ingest(request: SourceRequest) -> dict:
    """Persist pending chunks; exact repeat requests reuse document IDs."""
    chunks = _chunks(request)
    writer = dualwrite.write_pending_report if request.filename is not None else dualwrite.write_manual_report
    ids, inserted = await writer(chunks)
    if request.vectorize:
        try:
            vectorized = await _vectorize_pending()
        except Exception as exc:
            raise HTTPException(
                502, f"已入库 {len(ids)} 块（pending），向量化失败；补跑一次即可"
            ) from exc
    else:
        vectorized = 0
    return {
        "count": len(ids), "ids": ids, "inserted": inserted,
        "skipped": len(ids) - inserted, "vectorized": vectorized,
    }


async def _vectorize_pending() -> int:
    client = milvus_client.get_client()
    try:
        milvus_client.ensure_collection(client)
        return await dualwrite.vectorize_pending(client)
    finally:
        client.close()


@router.post("/vectorize")
async def vectorize() -> dict:
    """Complete pending MySQL→Milvus writes, closing the owned client."""
    try:
        done = await _vectorize_pending()
        return {"vectorized": done}
    except Exception as exc:
        raise HTTPException(502, "vectorization failed; pending chunks can be retried") from exc


@router.post("/search")
async def search(request: SearchRequest) -> dict:
    if not request.query.strip():
        raise HTTPException(400, "query is required")
    hits = await retrieval.search_knowledge(
        request.query, top_k=request.top_k, min_score=request.min_score,
    )
    return {"route": "dense", "hits": hits}
