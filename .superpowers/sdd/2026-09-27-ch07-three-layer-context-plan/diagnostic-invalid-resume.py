"""Diagnostic only: compare repeated invalid/valid resume against base runtime."""
import subprocess
import pytest
from sqlalchemy import text
from app.db import repository
from app.graph import nodes
from app.graph.build import build_graph
from app.graph.runtime import GraphRuntime
from app.api.chat import graph_event_stream

@pytest.mark.asyncio
@pytest.mark.parametrize('baseline', [True, False])
async def test_invalid_then_valid(tmp_path, db_session_factory, db_clean, monkeypatch, baseline):
    runtime_cls = GraphRuntime
    if baseline:
        namespace = {'__name__': 'baseline_runtime'}
        exec(__import__('pathlib').Path('.baseline_runtime.py').read_text(encoding='utf-8-sig'), namespace)
        runtime_cls = namespace['GraphRuntime']
    class Classifier:
        async def classify(self, query):
            return '退款退货'
    async def resolve(query, history, model):
        return query
    async def weak_policy(state, runtime):
        return {'sufficient': False, 'evidence': '', 'citations': [], 'reason': '无政策'}
    monkeypatch.setattr(nodes.coref_service, 'resolve', resolve)
    monkeypatch.setattr(nodes, 'retrieve_policy', weak_policy)
    async with db_session_factory.begin() as session:
        await session.execute(text("INSERT INTO sample_orders (order_id,user_id,status,product,amount) VALUES ('1001','owner','已签收','猫粮',88.00)"))
    async with runtime_cls(tmp_path / 'diag.sqlite', build_graph, classifier=Classifier()) as graph:
        first = await graph.ainvoke_turn('owner', '帮我退款', None, model=object())
        cid = first['conversation_id']
        bad = await graph.prepare_resume_turn('owner', cid, 'invalid', model=object())
        assert 'select_order' in ''.join([x async for x in graph_event_stream(bad, 'owner')])
        good = await graph.prepare_resume_turn('owner', cid, '1001', model=object())
        assert '[DONE]' in ''.join([x async for x in graph_event_stream(good, 'owner')])

