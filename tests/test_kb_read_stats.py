"""Exercise repository read queries against real SQLite SQL without MySQL service."""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.db import repository


@pytest.fixture
def read_database(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("""CREATE TABLE knowledge_chunks (
            id INTEGER PRIMARY KEY, category TEXT NOT NULL, questions TEXT NOT NULL,
            answer TEXT NOT NULL, section_path TEXT, content_type TEXT,
            is_key_clause INTEGER NOT NULL, prev_chunk_id INTEGER,
            next_chunk_id INTEGER, vector_id TEXT, vectorize_status TEXT NOT NULL,
            created_at TEXT, updated_at TEXT
        )"""))
        connection.execute(text("""CREATE TABLE qa_extraction_staging (
            id INTEGER PRIMARY KEY, batch_no TEXT NOT NULL, source_ref TEXT,
            question TEXT NOT NULL, answer TEXT NOT NULL, status TEXT NOT NULL,
            created_at TEXT
        )"""))

    class AsyncSessionAdapter:
        def __init__(self):
            self.session = Session(engine)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.session.close()

        async def execute(self, statement):
            return self.session.execute(statement)

    monkeypatch.setattr(repository.db, "async_session", AsyncSessionAdapter)
    yield engine
    engine.dispose()


@pytest.mark.asyncio
async def test_knowledge_stats_grouped_counts_and_empty_defaults(read_database):
    assert await repository.knowledge_stats() == {
        "total": 0, "pending": 0, "done": 0, "key_clauses": 0,
        "by_category": {}, "by_content_type": {},
    }
    with read_database.begin() as connection:
        connection.execute(text("""INSERT INTO knowledge_chunks
            (id,category,questions,answer,content_type,is_key_clause,vectorize_status)
            VALUES
            (1,'售后','退货','甲','policy',1,'pending'),
            (2,'售后','退款','乙','policy',0,'done'),
            (3,'物流','运费','丙',NULL,1,'done')"""))
    assert await repository.knowledge_stats() == {
        "total": 3, "pending": 1, "done": 2, "key_clauses": 2,
        "by_category": {"售后": 2, "物流": 1},
        "by_content_type": {"policy": 2, "unspecified": 1},
    }


@pytest.mark.asyncio
async def test_recent_chunks_reverse_id_and_limit(read_database):
    with read_database.begin() as connection:
        connection.execute(text("""INSERT INTO knowledge_chunks
            (id,category,questions,answer,is_key_clause,vectorize_status)
            VALUES (1,'a','q1','a1',0,'pending'), (2,'b','q2','a2',0,'done'),
                   (3,'c','q3','a3',0,'pending')"""))
    rows = await repository.list_recent_chunks(limit=2)
    assert [(row.id, row.vectorize_status) for row in rows] == [(3, "pending"), (2, "done")]
    assert await repository.list_recent_chunks(limit=0) == []


@pytest.mark.asyncio
async def test_chunk_pairs_preserve_distinct_answer_and_order(read_database):
    with read_database.begin() as connection:
        connection.execute(text("""INSERT INTO knowledge_chunks
            (id,category,questions,answer,is_key_clause,vectorize_status)
            VALUES (2,'a','same','second',0,'pending'),
                   (1,'a','same','first',0,'pending')"""))
    assert await repository.list_chunk_pairs() == [
        ("same", "first"), ("same", "second")
    ]


@pytest.mark.asyncio
async def test_staging_stats_three_states_and_distinct_batches(read_database):
    assert await repository.staging_stats() == {
        "extracted": 0, "kept": 0, "discarded": 0, "batches": 0,
    }
    with read_database.begin() as connection:
        connection.execute(text("""INSERT INTO qa_extraction_staging
            (id,batch_no,question,answer,status) VALUES
            (1,'one','q1','a1','extracted'), (2,'one','q2','a2','kept'),
            (3,'two','q3','a3','discarded'), (4,'two','q4','a4','kept')"""))
    assert await repository.staging_stats() == {
        "extracted": 1, "kept": 2, "discarded": 1, "batches": 2,
    }


@pytest.mark.asyncio
async def test_empty_extraction_marker_is_not_a_discarded_qa_pair(read_database):
    with read_database.begin() as connection:
        connection.execute(text("""INSERT INTO qa_extraction_staging
            (id,batch_no,question,answer,status) VALUES
            (1,'empty-source','','','discarded'),
            (2,'real-source','不适合入库','无法核实','discarded')"""))
    assert await repository.staging_stats() == {
        "extracted": 0, "kept": 0, "discarded": 1, "batches": 2,
    }
    assert [row.id for row in await repository.list_staging_by_status("discarded")] == [2]
