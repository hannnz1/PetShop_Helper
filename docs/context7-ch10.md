# 第 10 章接口核对（2026-09-28）

- Task 1 只使用 Python 标准库，本地术语表源于用户指定的课程仓库；无第三方 API 变更。
- Task 2 经 Context7 查询官方 [LangChain Python ChatOpenAI 文档](https://docs.langchain.com/oss/python/integrations/chat/openai)：`with_structured_output(PydanticModel, method=...)` 返回结构化对象；Prompt 用 `ChatPromptTemplate.from_messages`，异步执行用 `ainvoke`。本机锁定 `langchain-core 1.6.4`、`pydantic 2.13.5`；用假模型验证 Pydantic 结果与失败路径。
- Task 2 经 Context7 查询官方 [SQLAlchemy 2.0 文档](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)：`AsyncSession` 对 `select(...).limit(...)` 异步读取；本机锁定 `sqlalchemy 2.0.54`。低置信池只读并在本机脱敏。

后续 Transformers、PyTorch、ONNX Runtime、tokenizers、FastAPI 的具体实现必须按各任务再查 Context7 与锁定版本，不沿用课程旧代码的接口假设。
