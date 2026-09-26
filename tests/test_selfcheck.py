"""Flat evidence-sufficiency decision for the RAG pipeline."""

import pytest

from app.core import selfcheck


@pytest.mark.asyncio
async def test_check_sufficient_parses_flat_model_output(monkeypatch):
    class FakeChain:
        async def ainvoke(self, variables):
            assert variables == {
                "query": "Pro 型号功能",
                "evidence": "[1] 满99元包邮",
            }
            return type("Result", (), {"useful": False, "reason": "证据只讲运费，未涉及型号"})()

    monkeypatch.setattr(selfcheck, "_chain", FakeChain)
    result = await selfcheck.check_sufficient("Pro 型号功能", ["满99元包邮"])
    assert result == {"useful": False, "reason": "证据只讲运费，未涉及型号"}


@pytest.mark.asyncio
async def test_empty_evidence_rejects_without_model_call(monkeypatch):
    monkeypatch.setattr(selfcheck, "_chain", lambda: (_ for _ in ()).throw(AssertionError("model called")))
    assert await selfcheck.check_sufficient("保修期", []) == {
        "useful": False, "reason": "没有检索证据",
    }
