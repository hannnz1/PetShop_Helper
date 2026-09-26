# Ch07 Context7 API 核对（2026-09-27）

在编写第 7 章 spec 前通过 Context7 `resolve_library_id` 和 `query_docs` 查询官方文档。实现阶段若增加新的库 API 用法，先补查并追记到此文件。

| 库 | Context7 ID | 本章确认的接口 | 官方来源 |
| --- | --- | --- | --- |
| LangGraph | `/langchain-ai/langgraph` | `messages: Annotated[Sequence[BaseMessage], add_messages]` 让节点更新并入 State；编译图时传 checkpointer，按 thread ID 恢复完整状态 | [LangGraph 示例](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb)，[checkpoint README](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/README.md) |
| LangChain | `/websites/reference_langchain` | `trim_messages(..., max_tokens, token_counter, strategy="last", start_on="human", allow_partial=False)`；`count_tokens_approximately(..., chars_per_token=...)` 是估算而非上游精确计费值 | [trim_messages](https://reference.langchain.com/python/langchain-core/messages/utils/trim_messages)，[count_tokens_approximately](https://reference.langchain.com/python/langchain-core/messages/utils/count_tokens_approximately) |
| SQLAlchemy 2.0 | `/websites/sqlalchemy_en_20` | `AsyncSession.add()` 暂存记录，`await session.commit()` 提交；本仓库现用 `async_session.begin()` 把多行写入作为同一事务 | [AsyncIO ORM](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) |
| FastAPI | `/websites/fastapi_tiangolo` | `@app.get` 的异步读接口；`BackgroundTasks` 可在响应发出后运行任务，流式响应的完成和失败边界仍须按本仓库 SSE 流程验证 | [路径操作](https://fastapi.tiangolo.com)，[后台任务](https://fastapi.tiangolo.com/tutorial/background-tasks/) |

Context7 只确认 API 形态。分层边界、预算公式、事务并发与 SSE 生命周期属于本项目设计，需要通过独立测试验证。

## Task 2 implementation lookup

Before implementing the budgeter on 2026-09-27, rechecked the already resolved LangChain reference entry above. Its official [`count_tokens_approximately`](https://reference.langchain.com/python/langchain-core/messages/utils/count_tokens_approximately) API accepts `chars_per_token`; Task 2 passes the configured CJK calibration through that argument for both message estimates and representative-turn sizing. The result remains an estimate, not provider usage. `trim_messages` remains the existing final history gate with `allow_partial=False`; Task 2 adds no new library API.

## Task 1 implementation lookup

On 2026-09-27, resolved SQLAlchemy again to `/websites/sqlalchemy_en_20` before adding migration and boundary APIs. [AsyncIO ORM](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) confirms `AsyncConnection.execute(text(...), parameters)`, `async_sessionmaker.begin()` transactional sessions, and `AsyncSession.get(..., with_for_update=True)`. [SQL expression docs](https://docs.sqlalchemy.org/en/20/core/sqlelement.html) confirm named bound parameters in `text()`. These support schema inspection with bound database name and locking the conversation row before advancing an anchor.

## Task 3 implementation lookup

On 2026-09-27, resolved LangChain to `/websites/reference_langchain` and rechecked the official [`trim_messages`](https://reference.langchain.com/python/langchain-core/messages/utils/trim_messages) reference before implementing the pure context assembler. It confirms `strategy="last"`, a callable `token_counter`, `start_on="human"`, `end_on="ai"`, and `allow_partial=False`. The assembler uses the existing `trim_history` wrapper as its final whole-turn gate, so the checkpoint's original messages are untouched.

## Task 4 implementation lookup

On 2026-09-27, queried the resolved LangChain reference `/websites/reference_langchain` for model-bound tool schema measurement. The [BaseTool reference](https://reference.langchain.com/python/langchain-core/tools/base/BaseTool) identifies argument schemas and descriptions as tool metadata; the [Pydantic schema utility reference](https://reference.langchain.com/python/langchain-core/utils/pydantic/model_json_schema) confirms JSON schema generation. Task 4 measures the route's actual bound tools using `tool_call_schema.model_json_schema()` (already exercised by this repository's tool tests), including names and descriptions, before deriving the model budget. It still keeps a safety reserve for provider-specific serialization overhead.

For Task 4 review fix round 1, queried `/langchain-ai/langgraph` for [conditional edges](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py) and [runtime context](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/tests/test_runtime.py). `add_conditional_edges` accepts a callable path, and compiled graphs pass context to callables through `Runtime`; a compiled-graph regression verified runtime injection into this project's conditional router before retaining that implementation.

## Task 5 implementation lookup

On 2026-09-27, rechecked the resolved SQLAlchemy AsyncIO ORM reference already recorded above: `AsyncSession.get(..., with_for_update=True)` and `async_sessionmaker.begin()` keep the conversation row lock, new segment insert, and anchor update in one transaction. No new LangChain API is used beyond the previously checked `ChatPromptTemplate.format_messages()` and `BaseChatModel.ainvoke()`. For database guards, the official [MySQL trigger syntax](https://dev.mysql.com/doc/refman/8.4/en/trigger-syntax.html) documents `OLD`/`NEW` row values and `BEFORE UPDATE`/`BEFORE DELETE`; the official [SIGNAL statement](https://dev.mysql.com/doc/refman/8.4/en/signal.html) documents SQLSTATE `45000` for rejecting a write. The migration installs guards idempotently after the additive Ch07 schema.
