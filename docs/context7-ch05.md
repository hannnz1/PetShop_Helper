# 第 5 章 Context7 接口核对（2026-09-26）

使用 Context7 MCP 查询 `/langchain-ai/langgraph` 官方仓库文档和源码。首次两次连接失败，第三次恢复；下列结果来自成功查询，实施前仍须对已安装版本做本地冒烟。

| 主题 | 当前官方定义 | 来源 |
|---|---|---|
| 图与状态 | `StateGraph(State)`、`add_node`、`add_edge`、`add_conditional_edges`、`compile(checkpointer=...)`；`messages: Annotated[list[AnyMessage], add_messages]` | [LangGraph 官方仓库示例](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb)、[图构建示例](https://github.com/langchain-ai/langgraph/blob/main/examples/code_assistant/langgraph_code_assistant_mistral.ipynb) |
| 异步 SQLite 持久化 | `from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver`；`async with AsyncSqliteSaver.from_conn_string(path) as saver`；调用配置携带 `{"configurable": {"thread_id": ...}}` | [checkpoint-sqlite README](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-sqlite/README.md)、[checkpoint README](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/README.md) |
| 流式事件 | `astream(..., stream_mode=["messages", "updates"])` 在不启用子图流时返回 `(mode, payload)`；`messages` payload 是 `(message, metadata)`，metadata 可用于按节点过滤；`ainvoke(input, config=...)` 支持异步非流式调用 | [LangGraph pregel 源码](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py)、[messages 发射源码](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_messages.py) |

验证边界：Context7 文档说明接口形状，但不证明本机依赖安装、SQLite 文件权限、现有 FastAPI lifespan 或 SSE 事件接线正常。实施计划首个任务必须用已安装版本做无模型调用的 StateGraph + checkpointer + 消息/更新流冒烟。
