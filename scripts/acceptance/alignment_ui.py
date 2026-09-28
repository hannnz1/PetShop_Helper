"""Local UI acceptance fixture: real isolated MySQL, synthetic chat/vectors only.

Run on 127.0.0.1:8771 with DATABASE_URL naming mewhelp_alignment_ui_test.
No production schema is accepted and no model/embedding service is called.
"""

from contextlib import asynccontextmanager

from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.engine import make_url, URL
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.db import base, repository
from app.db.models import CanonicalQuestion
from app.main import create_app
from app.core.jobs import JOB_SPECS
from app.flywheel import review
from tests.conftest import _create_table_stmts


class FixtureGraph:
    async def prepare_stream_turn(self, user_id, message, conversation_id=None, **kwargs):
        if conversation_id is None:
            conversation_id = await repository.create_conversation(user_id)
        else:
            row = await repository.get_conversation(conversation_id)
            if row is None or row.user_id != user_id:
                from app.graph.runtime import ConversationNotFound
                raise ConversationNotFound()
        async def stream():
            answer = '这是隔离验收固定回复，未调用真实模型。问题：' + message
            yield 'updates', {'chitchat_reply': {'answer': answer}}
            marker = await repository.append_turn_messages(conversation_id, message, [], answer)
            yield 'updates', {'log_turn': {'conversation_id': conversation_id, 'trace': {'audit_message_id': marker}}}
        return stream()


class FixtureJobs:
    def list(self):
        return [{'name': name, 'state': 'fixture'} for name in JOB_SPECS]
    def status(self, name):
        return {'name': name, 'state': 'fixture', 'log_tail': '隔离验收禁止真实作业'}
    def start(self, name, **kwargs):
        return self.status(name)
    def stop(self, name):
        return self.status(name)


vector_attempts = 0


async def fixture_vectors():
    global vector_attempts
    vector_attempts += 1
    if vector_attempts == 1:
        raise ConnectionError('deliberate first vector failure')
    async with base.async_session.begin() as session:
        result = await session.execute(text("UPDATE knowledge_chunks SET vectorize_status='done' WHERE vectorize_status='pending'"))
        return result.rowcount


app = create_app()


@asynccontextmanager
async def lifespan(application):
    settings = get_settings()
    url = make_url(settings.database_url)
    if url.host != '127.0.0.1' or url.port != 3307 or url.database != 'mewhelp_alignment_ui_test':
        raise RuntimeError('UI fixture requires local 3307/mewhelp_alignment_ui_test')
    admin = create_async_engine(URL.create(url.drivername, username=url.username, password=url.password,
                                          host=url.host, port=url.port, query=url.query), isolation_level='AUTOCOMMIT')
    async with admin.connect() as conn:
        await conn.execute(text('CREATE DATABASE IF NOT EXISTS mewhelp_alignment_ui_test CHARACTER SET utf8mb4'))
    await admin.dispose()
    async with base.engine.begin() as conn:
        count = await conn.scalar(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='mewhelp_alignment_ui_test'"))
        if not count:
            for statement in _create_table_stmts():
                await conn.execute(text(statement))
    async with base.async_session.begin() as session:
        if not await session.scalar(text('SELECT COUNT(*) FROM canonical_questions')):
            session.add_all(CanonicalQuestion(canonical_question=question, draft_answer='仅供参考的模型草稿', status='pending_review')
                            for question in ['验收：猫粮可以退货吗？', '验收：补充售后信息', '验收：重复问题待合并', '验收：不相关问题待拒绝'])
    from scripts.ch10.migrate_topics import apply_topic_schema
    from app.db.models import LowConfidenceQuestion
    from app.db.topics import write_classifications
    from app.core.taxonomy import TOPIC_NAMES, terminology_table
    import hashlib
    await apply_topic_schema(base.engine)
    async with base.async_session.begin() as session:
        if not await session.scalar(text("SELECT COUNT(*) FROM low_confidence_questions WHERE source_ref LIKE 'alignment-topic-%'")):
            for index in range(22):
                session.add(LowConfidenceQuestion(raw_question=f'合成验收问题{index} 联系13800138000',
                    source='user_feedback', source_ref=f'alignment-topic-{index}'))
    async with base.async_session() as session:
        rows = (await session.execute(text("SELECT id,raw_question FROM low_confidence_questions WHERE source_ref LIKE 'alignment-topic-%' ORDER BY id"))).all()
    from scripts.ch10.corpus_lib import desensitize
    await write_classifications([{'id': row.id, 'text': desensitize(row.raw_question),
        'labels': list(TOPIC_NAMES[:2]) if index == 0 else [TOPIC_NAMES[0]]} for index,row in enumerate(rows)],
        {'model_version':'f'*64, 'taxonomy_hash':hashlib.sha256(terminology_table().encode()).hexdigest(), 'threshold':.5},
        'alignment-synthetic-classification')
    application.state.settings = settings
    application.state.jobs = FixtureJobs()
    application.state.graph = FixtureGraph()
    application.state.model = object()
    review.vectorize_pending_knowledge = fixture_vectors
    import app.api.chat as chat_api
    chat_api.schedule_summary = lambda *args, **kwargs: None
    yield
    await base.engine.dispose()


app.router.lifespan_context = lifespan
feedback_attempts = 0


@app.middleware('http')
async def deliberate_feedback_failure(request, call_next):
    global feedback_attempts
    if request.method == 'POST' and request.url.path == '/api/feedback/unresolved':
        feedback_attempts += 1
        if feedback_attempts == 1:
            return JSONResponse({'detail': '隔离验收：模拟网络失败，请重试'}, status_code=502)
    return await call_next(request)
