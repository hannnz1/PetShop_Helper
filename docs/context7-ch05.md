# 第 5 章 Context7 接口核对（2026-09-26）

使用 Context7 MCP 查询 `/langchain-ai/langgraph` 官方仓库文档和源码。首次两次连接失败，第三次恢复；下列结果来自成功查询，实施前仍须对已安装版本做本地冒烟。

| 主题 | 当前官方定义 | 来源 |
|---|---|---|
| 图与状态 | `StateGraph(State)`、`add_node`、`add_edge`、`add_conditional_edges`、`compile(checkpointer=...)`；`messages: Annotated[list[AnyMessage], add_messages]` | [LangGraph 官方仓库示例](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb)、[图构建示例](https://github.com/langchain-ai/langgraph/blob/main/examples/code_assistant/langgraph_code_assistant_mistral.ipynb) |
| 异步 SQLite 持久化 | `from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver`；`async with AsyncSqliteSaver.from_conn_string(path) as saver`；调用配置携带 `{"configurable": {"thread_id": ...}}` | [checkpoint-sqlite README](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-sqlite/README.md)、[checkpoint README](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/README.md) |
| 流式事件 | `astream(..., stream_mode=["messages", "updates"])` 在不启用子图流时返回 `(mode, payload)`；`messages` payload 是 `(message, metadata)`，metadata 可用于按节点过滤；`ainvoke(input, config=...)` 支持异步非流式调用 | [LangGraph pregel 源码](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py)、[messages 发射源码](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/_messages.py) |

验证边界：Context7 文档说明接口形状，但不证明本机依赖安装、SQLite 文件权限、现有 FastAPI lifespan 或 SSE 事件接线正常。实施计划首个任务必须用已安装版本做无模型调用的 StateGraph + checkpointer + 消息/更新流冒烟。

计划编写前又用 Context7 核对 `/websites/fastapi_tiangolo`：`FastAPI(lifespan=asynccontextmanager)` 的启动/清理边界、应用 `state`、`include_router`，以及异步生成器交给 `StreamingResponse` 的用法。来源：[生命周期](https://fastapi.tiangolo.com/advanced/events/)、[流式响应](https://fastapi.tiangolo.com/advanced/custom-response/)、[应用参考](https://fastapi.tiangolo.com/reference/fastapi/)。

SQLAlchemy 通过 Context7 `/websites/sqlalchemy_en_20` 核对 2.0 的 `AsyncSession`/事务及唯一约束冲突处理；同一个 `request_id` 的并发提交以数据库唯一约束为最终防线，捕获 `IntegrityError` 后须结束失败事务再查询已存在记录。来源：[AsyncIO 扩展](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)、[会话事务](https://docs.sqlalchemy.org/en/20/orm/session_transaction.html/)。实施前仍需在当前 MySQL/asyncmy 版本上做隔离集成测试。

Task 9 前补查 Context7 `/websites/langchain_oss_python`：`ChatPromptTemplate.from_messages(...) | model.with_structured_output(PydanticModel, method=...)` 可构成异步 `ainvoke` 链；OpenAI 集成官方示例使用 Pydantic schema 和 `json_schema`，当前项目已有 `json_mode` 配置与售后提取实测，故分类器沿用同一可配置 method。来源：[LangChain 模型结构化输出](https://docs.langchain.com/oss/python/integrations/chat/openai)、[Runnable 异步调用](https://docs.langchain.com/oss/python/integrations/chat/sambanova)。本机用 `RunnableLambda` 假模型验证链路，真实上游分类准确率待额度恢复。

代码复核修复前再查 Context7 `/langchain-ai/langgraph` 官方源码：编译图的 `aget_state(config)` 返回 `StateSnapshot`，缺检查点时 `values={}`、`next=()`；有未完成任务时 `next` 含后续节点。第 5 章运行时用它与 MySQL 最后审计消息 ID 比较，阻止缺失、过期或未完成检查点静默续聊。来源：[LangGraph Pregel 状态快照实现](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py)。
