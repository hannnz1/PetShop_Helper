# 第 6 章 Context7 官方接口核对（2026-09-26）

- LangGraph 官方 Python 文档 `/websites/langchain_oss_python_langgraph` 与 Python reference `/websites/reference_langchain_python_langgraph`：`interrupt(value)` 需要带 checkpointer 编译的图；中断流产生 `__interrupt__`，续跑使用相同 `thread_id` 的 `Command(resume=value)`；恢复会从被中断节点开头重跑。`PregelProtocol.astream` 接受 `Command` 输入及 `stream_mode` 列表。来源：[interrupt reference](https://reference.langchain.com/python/langgraph/types/interrupt)、[astream reference](https://reference.langchain.com/python/langgraph/pregel/protocol/PregelProtocol/astream)。真实 `messages+updates` chunk 形状仍由无模型本机冒烟确认。
- FastAPI 官方文档 `/websites/fastapi_tiangolo`：可返回包装异步生成器的 `StreamingResponse`；`HTTPException` 可在返回响应前终止请求。用于续跑预检先返回 404/409，再开始 SSE。来源：[StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/)、[HTTPException](https://fastapi.tiangolo.com/tutorial/handling-errors/)。
- SQLAlchemy 2.0 官方文档 `/websites/sqlalchemy_en_20`：`async_sessionmaker.begin()` 管理自动提交并关闭的异步事务；MySQL 唯一约束等同唯一索引。申请写入遇唯一冲突后必须离开失败事务再查同 `request_id`。来源：[AsyncIO ORM](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)、[MySQL dialect](https://docs.sqlalchemy.org/en/20/dialects/mysql.html)。
- LangChain Python 官方文档 `/websites/langchain_oss_python_langchain`：聊天模型可用消息对象调用，模型工具通过 `bind_tools` 产生结构化 `tool_calls`。本章 Prompt 链和建议动作工具沿用项目现有 LangChain 版本，首个测试验证实际接口。来源：[Models](https://docs.langchain.com/oss/python/langchain/models)。

政策查询扩写另行核对同一官方 Models 文档：`with_structured_output(PydanticModel, method="json_mode")` 支持运行时 Pydantic 校验；`json_mode` 只保证 JSON 形状，字段语义须在 Prompt 明示。本地配置仍使用 `structured_output_method`，没有改上游协议。

退款建议工具另查 `/websites/reference_langchain` 的 [`InjectedToolArg` reference](https://reference.langchain.com/python/langchain-core/tools/base/InjectedToolArg)：`Annotated` 标记的运行时注入参数不在模型可见 schema 中；本项目仍在图拦截层以可信 `state.user_id` 重验归属，通用执行分发层无条件拒绝同名调用。

以上查询不含密钥、用户数据或仓库专有代码。接口核对在实现前完成；不把文档示例等同本地版本行为。

## 本机安装版本冒烟

`scripts/smoke_ch06_interrupt.py` 在当前 `.venv` 无模型运行结果：`ainvoke` 返回 `__interrupt__[0].value.type=select_order`；`astream(stream_mode=["messages","updates"])` 的中断出现在 `updates` 负载的 `__interrupt__`；中断后 `aget_state().next == ("pick_order",)`。关闭并重开 `AsyncSqliteSaver` 后，相同 `thread_id` 的 `Command(resume="1001")` 返回 `order_id=1001`，`next` 变空，节点只读计数共 3 次（两条测试会话，含一次恢复重跑），恢复流产生 1 个消息块。对应测试 `tests/graph/test_interrupt_smoke.py` 通过。这是当前版本的实测形状，Task 7 按此解析。

Task 7 另核对官方 [`aget_state`](https://reference.langchain.com/python/langgraph/pregel/main/Pregel/aget_state) 与 [`StateSnapshot`](https://reference.langchain.com/python/langgraph/types/StateSnapshot)：本机待选单快照的 `next=("pick_order",)`（实际业务节点为 `fetch_order`）、`snapshot.interrupts[0].value.type="select_order"`，`tasks[0].interrupts` 有同一载荷。续跑前同时比较 MySQL 审计标记、节点名、载荷类型与会话归属；普通新消息在合法 pending 上返回 409。
