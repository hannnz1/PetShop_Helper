"""Offline proof of the installed LangGraph checkpoint and stream APIs."""

import asyncio
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class _State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    text: str


async def run_smoke(checkpoint_path: Path) -> dict[str, str | int]:
    """Run two turns on one thread; inspect the graph's actual stream tuples."""
    fake = FakeListChatModel(responses=["ok"])

    async def echo(state: _State) -> dict:
        current = next(
            message.content for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )
        async for _ in fake.astream([HumanMessage(content=str(current))]):
            pass
        text = " ".join(part for part in (state.get("text"), str(current)) if part)
        return {"text": text, "messages": [AIMessage(content=text)]}

    builder = StateGraph(_State)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    builder.add_conditional_edges("echo", lambda _: "done", {"done": END})

    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "ch05-smoke"}}
        first = await graph.ainvoke({"messages": [HumanMessage(content="first")]}, config)
        message_chunks = 0
        update_chunks = 0
        second = ""
        async for mode, payload in graph.astream(
            {"messages": [HumanMessage(content="second")]},
            config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                message, metadata = payload
                assert metadata["langgraph_node"] == "echo"
                assert isinstance(message, BaseMessage)
                message_chunks += 1
            elif mode == "updates":
                update_chunks += 1
                second = payload["echo"]["text"]
        return {
            "first": first["text"], "second": second,
            "message_chunks": message_chunks, "update_chunks": update_chunks,
        }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        print(asyncio.run(run_smoke(Path(directory) / "graph.sqlite")))
