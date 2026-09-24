# Ch02 Context7 MCP 核验记录

2026-09-24，在实施前通过 Context7 `resolve-library-id`/`query-docs` 查到以下官方文档；这是计划接口核验，实际实现仍需对照锁定版本和测试结果。

| 库 ID | 接口结论 | 官方出处 |
|---|---|---|
| `/websites/sqlalchemy_en_20` | `mysql+asyncmy://...?...charset=utf8mb4` 可传给 `create_async_engine()`；`async_sessionmaker` 生成可 `async with` 管理的 AsyncSession。 | https://docs.sqlalchemy.org/en/20/dialects/mysql.html 、https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html |
| `/websites/langchain_oss_python_langchain` | `@tool` 定义业务工具；`model.bind_tools([...])` 后响应的 `tool_calls` 含 `name`/`args`/`id`，工具结果以关联 `tool_call_id` 回灌。 | https://docs.langchain.com/oss/python/langchain/tools 、https://docs.langchain.com/oss/python/langchain/models |
| `/websites/reference_langchain` | `InjectedToolArg` 标注运行时注入参数，排除于提供给模型的工具 schema；计划需测试实际绑定 schema，不把可信会话 ID 暴露给模型。 | https://reference.langchain.com/python/langchain-core/tools/base/InjectedToolArg |

第 2 章实施必须使用用户固定的 SQLAlchemy 2.0 异步、asyncmy、Docker MySQL 与 LangChain 工具接口，不以 SQLite 或其他驱动替换。计划预审纠偏见 [ch02-execution-decisions.md](superpowers/plans/ch02-execution-decisions.md)。
