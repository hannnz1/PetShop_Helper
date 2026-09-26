"""Offline characterization of installed LangGraph interrupt and resume behavior."""

import asyncio
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt


class _State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    order_id: str


async def run_smoke(checkpoint_path: Path) -> dict:
    """Prove pause, durable resume, node rerun, and update/message stream shapes."""
    reads = 0

    async def pick_order(state: _State) -> dict:
        nonlocal reads
        reads += 1  # Diagnostic read counter; no business writes before interrupt.
        order_id = interrupt({"type": "select_order", "orders": ["1001"]})
        return {"order_id": order_id, "messages": [AIMessage(content=order_id)]}

    builder = StateGraph(_State)
    builder.add_node("pick_order", pick_order)
    builder.add_edge(START, "pick_order")
    config = {"configurable": {"thread_id": "ch06-invoke"}}

    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = builder.compile(checkpointer=saver)
        first = await graph.ainvoke({"messages": [HumanMessage(content="请退款")]}, config)
        pending_before = list((await graph.aget_state(config)).next)

    message_chunks = 0
    resumed_order = ""
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = builder.compile(checkpointer=saver)
        async for mode, payload in graph.astream(
            Command(resume="1001"), config, stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                message, metadata = payload
                if metadata.get("langgraph_node") == "pick_order" and isinstance(message, BaseMessage):
                    message_chunks += 1
            elif mode == "updates" and "pick_order" in payload:
                resumed_order = payload["pick_order"]["order_id"]
        pending_after = list((await graph.aget_state(config)).next)

        stream_kind = ""
        async for mode, payload in graph.astream(
            {"messages": [HumanMessage(content="另一会话")]},
            {"configurable": {"thread_id": "ch06-stream"}},
            stream_mode=["messages", "updates"],
        ):
            if mode == "updates" and "__interrupt__" in payload:
                stream_kind = payload["__interrupt__"][0].value["type"]

    return {
        "invoke_kind": first["__interrupt__"][0].value["type"],
        "stream_kind": stream_kind,
        "pending_before": pending_before,
        "pending_after": pending_after,
        "resumed_order": resumed_order,
        "reads": reads,
        "message_chunks": message_chunks,
    }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        print(asyncio.run(run_smoke(Path(directory) / "interrupt.sqlite")))
