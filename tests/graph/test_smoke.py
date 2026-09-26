"""Exercise the installed LangGraph APIs without any model calls."""

import pytest


@pytest.mark.asyncio
async def test_graph_sqlite_checkpoint_and_stream_shapes(tmp_path):
    from scripts.smoke_ch05_graph import run_smoke

    result = await run_smoke(tmp_path / "graph-checkpoints.sqlite")
    assert result["first"] == "first"
    assert result["second"] == "first second"
    assert result["message_chunks"] >= 1
    assert result["update_chunks"] >= 1
