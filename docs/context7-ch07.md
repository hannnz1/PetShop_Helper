# Ch07 Context7 API 核对（2026-09-27）

在编写第 7 章 spec 前通过 Context7 `resolve_library_id` 和 `query_docs` 查询官方文档。实现阶段若增加新的库 API 用法，先补查并追记到此文件。

| 库 | Context7 ID | 本章确认的接口 | 官方来源 |
| --- | --- | --- | --- |
| LangGraph | `/langchain-ai/langgraph` | `messages: Annotated[Sequence[BaseMessage], add_messages]` 让节点更新并入 State；编译图时传 checkpointer，按 thread ID 恢复完整状态 | [LangGraph 示例](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb)，[checkpoint README](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/README.md) |
| LangChain | `/websites/reference_langchain` | `trim_messages(..., max_tokens, token_counter, strategy="last", start_on="human", allow_partial=False)`；`count_tokens_approximately(..., chars_per_token=...)` 是估算而非上游精确计费值 | [trim_messages](https://reference.langchain.com/python/langchain-core/messages/utils/trim_messages)，[count_tokens_approximately](https://reference.langchain.com/python/langchain-core/messages/utils/count_tokens_approximately) |
| SQLAlchemy 2.0 | `/websites/sqlalchemy_en_20` | `AsyncSession.add()` 暂存记录，`await session.commit()` 提交；本仓库现用 `async_session.begin()` 把多行写入作为同一事务 | [AsyncIO ORM](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) |
| FastAPI | `/websites/fastapi_tiangolo` | `@app.get` 的异步读接口；`BackgroundTasks` 可在响应发出后运行任务，流式响应的完成和失败边界仍须按本仓库 SSE 流程验证 | [路径操作](https://fastapi.tiangolo.com)，[后台任务](https://fastapi.tiangolo.com/tutorial/background-tasks/) |

Context7 只确认 API 形态。分层边界、预算公式、事务并发与 SSE 生命周期属于本项目设计，需要通过独立测试验证。
