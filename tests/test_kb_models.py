"""Chapter-three ORM mappings against the isolated MySQL test schema."""

import pytest
from sqlalchemy import select

from app.db.models import KnowledgeChunk, QaExtractionStaging


@pytest.mark.asyncio
async def test_knowledge_chunk_roundtrip(db_session_factory, db_clean):
    async with db_session_factory() as session:
        first = KnowledgeChunk(category="售后政策", questions="运费说明", answer="满99包邮")
        session.add(first)
        await session.commit()
        await session.refresh(first)
        assert first.id is not None
        assert first.vectorize_status == "pending"
        assert first.is_key_clause == 0

        second = KnowledgeChunk(
            category="售后政策", questions="偏远地区", answer="按页面显示计费",
            section_path="配送/运费", content_type="policy", is_key_clause=1,
            prev_chunk_id=first.id,
        )
        session.add(second)
        await session.commit()
        got = (await session.execute(select(KnowledgeChunk).order_by(KnowledgeChunk.id))).scalars().all()
        assert len(got) == 2
        assert got[1].prev_chunk_id == first.id
        assert got[1].section_path == "配送/运费"
        assert got[1].content_type == "policy"


@pytest.mark.asyncio
async def test_staging_roundtrip(db_session_factory, db_clean):
    async with db_session_factory() as session:
        row = QaExtractionStaging(batch_no="b1", source_ref="conv-1", question="邮费多少", answer="满99包邮")
        session.add(row)
        await session.commit()
        got = (await session.execute(select(QaExtractionStaging))).scalars().all()
        assert len(got) == 1
        assert got[0].id is not None
        assert got[0].status == "extracted"
        assert got[0].source_ref == "conv-1"
