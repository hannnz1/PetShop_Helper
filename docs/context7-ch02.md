# Ch02 Context7 MCP 核验记录

2026-09-24，在实施前通过 Context7 `resolve-library-id`/`query-docs` 查到以下官方文档；这是计划接口核验，实际实现仍需对照锁定版本和测试结果。

| 库 ID | 接口结论 | 官方出处 |
|---|---|---|
| `/websites/sqlalchemy_en_20` | `mysql+asyncmy://...?...charset=utf8mb4` 可传给 `create_async_engine()`；`async_sessionmaker` 生成可 `async with` 管理的 AsyncSession。 | https://docs.sqlalchemy.org/en/20/dialects/mysql.html 、https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html |
| `/websites/langchain_oss_python_langchain` | `@tool` 定义业务工具；`model.bind_tools([...])` 后响应的 `tool_calls` 含 `name`/`args`/`id`，工具结果以关联 `tool_call_id` 回灌。 | https://docs.langchain.com/oss/python/langchain/tools 、https://docs.langchain.com/oss/python/langchain/models |
| `/websites/reference_langchain` | `InjectedToolArg` 标注运行时注入参数，排除于提供给模型的工具 schema；计划需测试实际绑定 schema，不把可信会话 ID 暴露给模型。 | https://reference.langchain.com/python/langchain-core/tools/base/InjectedToolArg |

第 2 章实施必须使用用户固定的 SQLAlchemy 2.0 异步、asyncmy、Docker MySQL 与 LangChain 工具接口，不以 SQLite 或其他驱动替换。计划预审纠偏见 [ch02-execution-decisions.md](superpowers/plans/ch02-execution-decisions.md)。

## Task 9：工具执行边界与 FAQ 上限

2026-09-25 实施前通过 Context7 再查官方接口：LangChain [ToolMessage](https://reference.langchain.com/python/langchain-core/messages/tool/ToolMessage) 支持关联 `tool_call_id`，错误消息可用 `status="error"`；[BaseTool.ainvoke](https://reference.langchain.com/python/langchain-core/tools/base/BaseTool) 接受工具输入并异步执行。SQLAlchemy [连接池断线处理](https://docs.sqlalchemy.org/en/20/core/pooling.html) 说明数据库断线经 `DBAPIError` 传播，连接会被失效；[ORM 查询](https://docs.sqlalchemy.org/en/20/orm/queryguide/select.html) 使用 `select(...).order_by(...)`，`.limit(10)` 对应 SQL 层结果上限。实际执行层以 `InjectedToolArg` 保留模型不可见的 `conversation_id`，并覆盖任何模型自带同名值；读工具只有限次重试，写工具从不自动重试；数据库类异常及 DB 工具超时留给上层接口处理。

## Task 3：ORM 映射补充核验

2026-09-24 再次通过 Context7 `/websites/sqlalchemy_en_20` 核对 SQLAlchemy 2 映射和 MySQL 方言。`Mapped`/`mapped_column` 是声明式映射接口；MySQL 方言 `BIGINT(unsigned=True)` 和 `ENUM(...)` 分别对应 DDL 的无符号整数、原生枚举。MySQL 的 `ON UPDATE CURRENT_TIMESTAMP` **不会**由 `server_onupdate` 自动生成，映射需用 `server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")` 表示其 DDL，并以 `server_onupdate=FetchedValue()` 通知 ORM 值由服务器更新。参见 [声明式表映射](https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html) 与 [MySQL 方言说明](https://docs.sqlalchemy.org/en/20/dialects/mysql.html)。Task 3 的实际表仍以 `sql/ch02-ddl.sql` 为准，映射不调用 `metadata.create_all`。
