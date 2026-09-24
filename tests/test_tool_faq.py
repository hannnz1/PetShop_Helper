"""FAQ tool behavior against the isolated Docker MySQL test schema."""

import pytest

from app.db.models import Faq
from app.tools.business import query_faq


@pytest.mark.asyncio
async def test_query_faq_hit_returns_question_and_answer(db_session_factory, db_clean):
    async with db_session_factory() as session:
        session.add(Faq(question="退货政策", answer="7 天无理由退货", category="售后"))
        await session.commit()

    result = await query_faq.ainvoke({"keyword": "退货政策"})

    assert result == {"hits": [{"question": "退货政策", "answer": "7 天无理由退货"}]}


@pytest.mark.asyncio
async def test_query_faq_literal_miss_exposes_synonym_gap(db_session_factory, db_clean):
    async with db_session_factory() as session:
        session.add(Faq(question="运费怎么算", answer="按地址计算", category="物流"))
        await session.commit()

    result = await query_faq.ainvoke({"keyword": "邮费"})

    assert result["hits"] == []
    assert "未找到" in result["message"]
    assert "邮费" in result["message"]
