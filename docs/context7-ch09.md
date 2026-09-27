# 第 9 章 Context7 核对记录

2026-09-27。实现具体库接口前继续按所装版本查询并验证。本次 spec 阶段已查：

- `/langfuse/langfuse-docs`：当前 [LangGraph 集成文档](https://github.com/langfuse/langfuse-docs/blob/main/content/integrations/frameworks/langgraph.mdx) 使用 `from langfuse.langchain import CallbackHandler`，在 `graph.stream(..., config={"callbacks": [handler]})` 或编译后 `with_config` 附加回调。设计选每轮运行配置以覆盖 Graph 的调用、流式和恢复入口。
- `/langfuse/langfuse-docs`：[环境变量与掩码文档](https://github.com/langfuse/langfuse-docs/blob/main/content/docs/observability/features/masking.mdx) 提醒 SDK `mask` 不会自动审查第三方 OpenTelemetry span 属性；不能把“已设置 mask”当作所有敏感字段均已过滤的证明。
- `/websites/langfuse_self-hosting`：[Docker Compose 部署](https://langfuse.com/self-hosting/deployment/docker-compose) 为本地/小规模部署推荐路径；[容量文档](https://langfuse.com/self-hosting/scaling) 列出 Web/Worker/PostgreSQL、Redis 和 ClickHouse 的内存要求。当前本机 Docker 报告约 20 GiB 可用内存，启动前仍需实测。

不要从旧教程照抄 `langfuse.callback` 路径或声称“编译时挂一次回调”必然覆盖当前全部调用；以后续 Context7 与实装版本试验为准。
