"""Source identities and fixture coverage for offline knowledge ingestion."""

import pytest

from app.db import repository
from app.kb import dualwrite
from scripts import build_kb
from scripts.build_kb import KB_DIR, DOCS, source_chunks


def test_source_fixtures_have_shipping_and_explicit_key_clause():
    chunks_by_file = {
        filename: source_chunks(filename, (KB_DIR / filename).read_text(encoding="utf-8"), kind)
        for filename, kind in DOCS.items()
    }
    all_chunks = [chunk for chunks in chunks_by_file.values() for chunk in chunks]
    faq_questions = {chunk.questions for chunk in chunks_by_file["product-faq.md"]}
    assert len(faq_questions) == 6
    assert any(chunk.questions == "运费怎么算" and "包邮" in chunk.answer for chunk in all_chunks)
    assert any(chunk.is_key_clause for chunk in all_chunks)
    assert all("key-clause" not in chunk.answer for chunk in all_chunks)
    manual_table = [
        chunk for chunk in chunks_by_file["after-sales-manual.md"]
        if chunk.questions == "常见问题处理时限"
    ]
    assert len(manual_table) == 2
    assert all(chunk.answer.startswith("| 问题类型 | 首次响应 | 处理时限 |") for chunk in manual_table)


@pytest.mark.asyncio
async def test_same_content_from_different_sources_remains_distinct(db_session_factory, db_clean):
    text = "# 相同手册\n\n## 相同章节\n\n同一正文。"
    a = source_chunks("first.md", text, "manual")
    b = source_chunks("second.md", text, "manual")
    first = await dualwrite.write_pending(a)
    second = await dualwrite.write_pending(b)
    assert first != second
    assert await dualwrite.write_pending(a) == first
    assert await repository.count_chunks_by_status("pending") == 2


@pytest.mark.asyncio
async def test_same_source_and_section_with_changed_body_keeps_new_version(db_session_factory, db_clean):
    first = source_chunks("policy.md", "# A\n\n## B\n\n旧正文。", "policy")
    changed = source_chunks("policy.md", "# A\n\n## B\n\n新正文。", "policy")
    old_ids = await dualwrite.write_pending(first)
    new_ids = await dualwrite.write_pending(changed)
    assert old_ids != new_ids
    assert await dualwrite.write_pending(changed) == new_ids
    assert await repository.count_chunks_by_status("pending") == 2


@pytest.mark.asyncio
async def test_build_cli_twice_is_idempotent(db_session_factory, db_clean, tmp_path, monkeypatch):
    (tmp_path / "fixture.md").write_text("# A\n\n## B\n\n同一正文。", encoding="utf-8")
    monkeypatch.setattr(build_kb, "KB_DIR", tmp_path)
    monkeypatch.setattr(build_kb, "DOCS", {"fixture.md": "policy"})
    await build_kb.main()
    assert await repository.count_chunks_by_status("pending") == 1
    await build_kb.main()
    assert await repository.count_chunks_by_status("pending") == 1


def test_source_path_rejects_traversal():
    with pytest.raises(ValueError, match="source"):
        source_chunks("../outside.md", "# A\n\ntext", "manual")
