"""Manual-ingest SQL behavior without the unavailable Docker MySQL engine."""

import pytest
from sqlalchemy import create_engine, text

from app.db import repository


class _AsyncTransaction:
    def __init__(self, transaction):
        self.transaction = transaction

    async def __aenter__(self):
        return self

    async def __aexit__(self, error_type, error, traceback):
        if error_type is None:
            self.transaction.commit()
        else:
            self.transaction.rollback()


class _AsyncConnection:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.connection.close()

    async def scalar(self, statement, parameters=None):
        sql = str(statement)
        if "GET_LOCK" in sql or "RELEASE_LOCK" in sql:
            return 1
        return self.connection.scalar(statement, parameters)

    async def execute(self, statement, parameters=None):
        return self.connection.execute(statement, parameters or {})

    async def commit(self):
        self.connection.commit()

    def begin(self):
        return _AsyncTransaction(self.connection.begin())


class _AsyncEngine:
    def __init__(self, engine):
        self.engine = engine
        self.url = engine.url

    def connect(self):
        return _AsyncConnection(self.engine.connect())


@pytest.fixture
def manual_database(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("""CREATE TABLE knowledge_chunks (
            id INTEGER PRIMARY KEY, category TEXT NOT NULL, questions TEXT NOT NULL,
            answer TEXT NOT NULL, section_path TEXT, content_type TEXT,
            is_key_clause INTEGER NOT NULL DEFAULT 0, prev_chunk_id INTEGER,
            next_chunk_id INTEGER, vector_id TEXT,
            vectorize_status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT, updated_at TEXT
        )"""))
        connection.execute(text("""CREATE TABLE qa_extraction_staging (
            id INTEGER PRIMARY KEY, batch_no TEXT NOT NULL, source_ref TEXT,
            question TEXT NOT NULL, answer TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'extracted', created_at TEXT
        )"""))
    monkeypatch.setattr(repository.db, "async_session", type("Session", (), {
        "kw": {"bind": _AsyncEngine(engine)},
    })())
    yield engine
    engine.dispose()


def _chunk(question, answer):
    return {
        "category": "演示", "questions": question, "answer": answer,
        "section_path": "manual :: 演示", "content_type": "policy",
        "is_key_clause": 0,
    }


@pytest.mark.asyncio
async def test_manual_dedup_keeps_same_heading_different_answers(manual_database):
    first, inserted = await repository.insert_manual_knowledge_report([
        _chunk("处理时限", "首次响应 24 小时"),
        _chunk("处理时限", "售后完成 3 天"),
    ])
    assert inserted == 2
    assert len(set(first)) == 2
    repeated, inserted = await repository.insert_manual_knowledge_report([
        _chunk("处理时限？", "首次响应24小时。"),
        _chunk("处理时限", "售后完成3天"),
    ])
    assert repeated == first
    assert inserted == 0
    changed, inserted = await repository.insert_manual_knowledge_report([
        _chunk("处理时限", "售后完成 5 天"),
    ])
    assert inserted == 1
    assert changed[0] not in first
    with manual_database.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM knowledge_chunks")) == 3


@pytest.mark.asyncio
async def test_manual_ingest_rolls_back_entire_document_on_insert_failure(manual_database, monkeypatch):
    original = repository._insert_document_row
    calls = 0

    async def fail_second(connection, chunk):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected interruption")
        return await original(connection, chunk)

    monkeypatch.setattr(repository, "_insert_document_row", fail_second)
    with pytest.raises(RuntimeError, match="injected interruption"):
        await repository.insert_manual_knowledge_report([
            _chunk("同节", "正文 A"), _chunk("同节", "正文 B"),
        ])
    with manual_database.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM knowledge_chunks")) == 0
    monkeypatch.setattr(repository, "_insert_document_row", original)
    ids, inserted = await repository.insert_manual_knowledge_report([
        _chunk("同节", "正文 A"), _chunk("同节", "正文 B"),
    ])
    assert len(ids) == inserted == 2


@pytest.mark.asyncio
async def test_manual_reuses_existing_pair_without_relinking_old_document(manual_database):
    with manual_database.begin() as connection:
        connection.execute(text("""INSERT INTO knowledge_chunks
            (id, category, questions, answer, section_path, content_type,
             is_key_clause, prev_chunk_id, next_chunk_id)
            VALUES (7, '原文档', '运费', '满99元包邮', 'source :: 运费', 'faq', 0, NULL, 8),
                   (8, '原文档', '退款', '联系客服', 'source :: 退款', 'faq', 0, 7, NULL)
        """))
    ids, inserted = await repository.insert_manual_knowledge_report([
        _chunk("运费？", "满99元包邮。"), _chunk("保修", "保修期一年"),
    ])
    assert ids[0] == 7
    assert inserted == 1
    with manual_database.connect() as connection:
        old = connection.execute(text(
            "SELECT prev_chunk_id, next_chunk_id FROM knowledge_chunks WHERE id = 7"
        )).one()
        assert old == (None, 8)
        assert connection.scalar(text("SELECT count(*) FROM knowledge_chunks")) == 3


@pytest.mark.asyncio
async def test_reset_drops_vectors_then_clears_only_chapter_three_tables(manual_database):
    with manual_database.begin() as connection:
        connection.execute(text("""INSERT INTO knowledge_chunks
            (id, category, questions, answer, is_key_clause)
            VALUES (1, '演示', 'q', 'a', 0)"""))
        connection.execute(text("""INSERT INTO qa_extraction_staging
            (id, batch_no, question, answer) VALUES (1, 'b', 'q', 'a')"""))
    called = []
    await repository.reset_knowledge_tables(lambda: called.append("dropped"))
    assert called == ["dropped"]
    with manual_database.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM knowledge_chunks")) == 0
        assert connection.scalar(text("SELECT count(*) FROM qa_extraction_staging")) == 0


@pytest.mark.asyncio
async def test_reset_keeps_mysql_if_vector_drop_fails(manual_database):
    with manual_database.begin() as connection:
        connection.execute(text("""INSERT INTO knowledge_chunks
            (id, category, questions, answer, is_key_clause)
            VALUES (1, '演示', 'q', 'a', 0)"""))

    def broken_drop():
        raise RuntimeError("Milvus offline")

    with pytest.raises(RuntimeError, match="Milvus offline"):
        await repository.reset_knowledge_tables(broken_drop)
    with manual_database.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM knowledge_chunks")) == 1
