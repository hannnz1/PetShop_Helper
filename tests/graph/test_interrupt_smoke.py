"""Characterize this installation's durable human-interrupt stream contract."""

import pytest


@pytest.mark.asyncio
async def test_interrupt_resume_survives_reopen_and_streams(tmp_path):
    from scripts.smoke_ch06_interrupt import run_smoke

    result = await run_smoke(tmp_path / "interrupt.sqlite")
    assert result["invoke_kind"] == "select_order"
    assert result["stream_kind"] == "select_order"
    assert result["pending_before"] == ["pick_order"]
    assert result["pending_after"] == []
    assert result["resumed_order"] == "1001"
    assert result["reads"] >= 2
    assert result["message_chunks"] >= 1
