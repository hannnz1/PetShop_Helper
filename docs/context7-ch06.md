# 第 6 章 Context7 官方接口核对（2026-09-26）

- LangGraph 官方 Python 文档 `/websites/langchain_oss_python_langgraph` 与 Python reference `/websites/reference_langchain_python_langgraph`：`interrupt(value)` 需要带 checkpointer 编译的图；中断流产生 `__interrupt__`，续跑使用相同 `thread_id` 的 `Command(resume=value)`；恢复会从被中断节点开头重跑。`PregelProtocol.astream` 接受 `Command` 输入及 `stream_mode` 列表。来源：[interrupt reference](https://reference.langchain.com/python/langgraph/types/interrupt)、[astream reference](https://reference.langchain.com/python/langgraph/pregel/protocol/PregelProtocol/astream)。真实 `messages+updates` chunk 形状仍由无模型本机冒烟确认。
- FastAPI 官方文档 `/websites/fastapi_tiangolo`：可返回包装异步生成器的 `StreamingResponse`；`HTTPException` 可在返回响应前终止请求。用于续跑预检先返回 404/409，再开始 SSE。来源：[StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/)、[HTTPException](https://fastapi.tiangolo.com/tutorial/handling-errors/)。
- SQLAlchemy 2.0 官方文档 `/websites/sqlalchemy_en_20`：`async_sessionmaker.begin()` 管理自动提交并关闭的异步事务；MySQL 唯一约束等同唯一索引。申请写入遇唯一冲突后必须离开失败事务再查同 `request_id`。来源：[AsyncIO ORM](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)、[MySQL dialect](https://docs.sqlalchemy.org/en/20/dialects/mysql.html)。
- LangChain Python 官方文档 `/websites/langchain_oss_python_langchain`：聊天模型可用消息对象调用，模型工具通过 `bind_tools` 产生结构化 `tool_calls`。本章 Prompt 链和建议动作工具沿用项目现有 LangChain 版本，首个测试验证实际接口。来源：[Models](https://docs.langchain.com/oss/python/langchain/models)。

以上查询不含密钥、用户数据或仓库专有代码。接口核对在实现前完成；不把文档示例等同本地版本行为。

## 本机安装版本冒烟

`scripts/smoke_ch06_interrupt.py` 在当前 `.venv` 无模型运行结果：`ainvoke` 返回 `__interrupt__[0].value.type=select_order`；`astream(stream_mode=["messages","updates"])` 的中断出现在 `updates` 负载的 `__interrupt__`；中断后 `aget_state().next == ("pick_order",)`。关闭并重开 `AsyncSqliteSaver` 后，相同 `thread_id` 的 `Command(resume="1001")` 返回 `order_id=1001`，`next` 变空，节点只读计数共 3 次（两条测试会话，含一次恢复重跑），恢复流产生 1 个消息块。对应测试 `tests/graph/test_interrupt_smoke.py` 通过。这是当前版本的实测形状，Task 7 按此解析。
