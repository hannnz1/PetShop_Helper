# 第 9 章第二部分 Context7 核对

2026-09-27，在书面设计前按项目要求查询当前官方文档；真正实现具体接口前仍须核对本机锁定版本的签名。

- Context7 `/websites/langchain_oss_python`：[LangChain Python ChatOpenAI 文档](https://docs.langchain.com/oss/python/integrations/chat/openai)示例使用 `with_structured_output(PydanticModel, method="json_schema")`，返回经过 Pydantic 校验的对象。其他 OpenAI 兼容上游未必支持该模式，设计要求能力失败时标记待处理，不伪造标准化结果或私自换模型。
- Context7 `/websites/sqlalchemy_en_20`：[SQLAlchemy asyncio 文档](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)中 `async_sessionmaker.begin()` 为自动提交事务上下文；[行锁 API](https://docs.sqlalchemy.org/en/20/core/selectable.html)提供 `select(...).with_for_update()`。去重合并须在同一事务中验证候选、写关联与计数，数据库唯一约束兜底并发重试。
- Context7 `/websites/fastapi_tiangolo`：[FastAPI HTTPBearer 参考](https://fastapi.tiangolo.com/reference/security)通过依赖提取 Bearer 凭据，`auto_error=False` 可由应用统一控制未配置/无凭据/错误凭据的响应。人工审核路由使用独立配置的令牌，不继承现有无鉴权知识管理接口。

本章不改动已锁定的 FastAPI、LangChain、MySQL、Milvus 选型。真实用户原话发送到外部模型前须另有授权；离线验收使用合成样例。
