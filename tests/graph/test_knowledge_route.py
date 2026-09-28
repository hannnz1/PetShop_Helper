"""Knowledge questions must pass the Chapter 4 evidence gate before generation."""

import pytest


@pytest.mark.asyncio
async def test_strong_numbered_evidence_enters_agent_without_low_confidence_write(monkeypatch):
    from app.graph import nodes

    written = []

    class Faq:
        async def ainvoke(self, args):
            assert args == {"keyword": "MH-W40 滤芯多久换"}
            return {"sufficient": True, "evidence": "[1] MH-W40: 每30天更换",
                    "citations": [{"n": 1, "id": 19, "section_path": "MH-W40",
                                   "question": "滤芯多久换", "answer": "每30天更换"}]}

    async def save(*args):
        written.append(args)

    async def pipeline(query, *, gate):
        assert gate == 'calibrated'
        return await Faq().ainvoke({'keyword': query})

    monkeypatch.setattr(nodes, 'run_faq_pipeline', pipeline)
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", save)
    update = await nodes.forced_rag({"query": "MH-W40 滤芯多久换", "conversation_id": 17})
    assert update["sufficient"] is True
    assert update["citations"][0]["n"] == 1
    assert nodes.confidence_gate(update) == "agent"
    assert written == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "source"),
    [({"sufficient": False, "source": "retrieval_low_conf", "reason": "top=0.1", "citations": []}, "retrieval_low_conf"),
     ({"sufficient": False, "source": "self_check", "reason": "证据不够", "citations": []}, "self_check"),
     ({"hits": [{"question": "旧格式", "answer": "不可信"}]}, "self_check")],
)
async def test_weak_or_legacy_result_refuses_and_records_one_question(monkeypatch, payload, source):
    from app.graph import nodes

    written = []

    class Faq:
        async def ainvoke(self, args):
            return payload

    async def save(*args):
        written.append(args)

    async def pipeline(query, *, gate):
        assert gate == 'calibrated'
        return await Faq().ainvoke({'keyword': query})

    monkeypatch.setattr(nodes, 'run_faq_pipeline', pipeline)
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", save)
    update = await nodes.forced_rag({"query": "火星车有货吗", "conversation_id": 17})
    assert update["sufficient"] is False
    assert update["citations"] == [] and update["evidence"] == ""
    assert nodes.confidence_gate(update) == "fallback"
    assert len(written) == 1
    assert written[0][:3] == (17, "火星车有货吗", source)


@pytest.mark.asyncio
async def test_retrieval_error_fails_closed_and_records_reason(monkeypatch):
    from app.graph import nodes

    written = []

    class Faq:
        async def ainvoke(self, args):
            raise TimeoutError("provider down")

    async def save(*args):
        written.append(args)

    async def pipeline(query, *, gate):
        assert gate == 'calibrated'
        return await Faq().ainvoke({'keyword': query})

    monkeypatch.setattr(nodes, 'run_faq_pipeline', pipeline)
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", save)
    update = await nodes.forced_rag({"query": "退款规定", "conversation_id": 17})
    assert update["sufficient"] is False
    assert nodes.confidence_gate(update) == "fallback"
    assert len(written) == 1
    assert "TimeoutError" in written[0][3]
