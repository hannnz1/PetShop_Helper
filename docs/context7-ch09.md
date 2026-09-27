# 第 9 章 Context7 核对记录

2026-09-27。实现具体库接口前继续按所装版本查询并验证。本次 spec 阶段已查：

- `/langfuse/langfuse-docs`：当前 [LangGraph 集成文档](https://github.com/langfuse/langfuse-docs/blob/main/content/integrations/frameworks/langgraph.mdx) 使用 `from langfuse.langchain import CallbackHandler`，在 `graph.stream(..., config={"callbacks": [handler]})` 或编译后 `with_config` 附加回调。设计选每轮运行配置以覆盖 Graph 的调用、流式和恢复入口。
- `/langfuse/langfuse-docs`：[环境变量与掩码文档](https://github.com/langfuse/langfuse-docs/blob/main/content/docs/observability/features/masking.mdx) 提醒 SDK `mask` 不会自动审查第三方 OpenTelemetry span 属性；不能把“已设置 mask”当作所有敏感字段均已过滤的证明。
- `/websites/langfuse_self-hosting`：[Docker Compose 部署](https://langfuse.com/self-hosting/deployment/docker-compose) 为本地/小规模部署推荐路径；[容量文档](https://langfuse.com/self-hosting/scaling) 列出 Web/Worker/PostgreSQL、Redis 和 ClickHouse 的内存要求。当前本机 Docker 报告约 20 GiB 可用内存，启动前仍需实测。
- `/langfuse/langfuse-python`：当前 SDK `CallbackHandler` 位于 `langfuse.langchain`，支持从环境变量读取 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL`；[回调源码](https://github.com/langfuse/langfuse-python/blob/main/langfuse/langchain/CallbackHandler.py) 从 RunnableConfig metadata 读取 `langfuse_session_id`、`langfuse_user_id`、`langfuse_trace_name`。
- `/websites/reference_langchain`：[用量回调参考](https://reference.langchain.com/python/langchain-core/callbacks/usage/UsageMetadataCallbackHandler) 使用 `AIMessage.usage_metadata`；流式 ChatOpenAI 的 [stream_usage](https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI) 可附加 token 用量。
- [Langfuse 官方发布列表](https://github.com/langfuse/langfuse/releases) 在 2026-09-27 显示自托管 v4.46.0 为最近发布版；执行部署前再次核实对应 Compose 和镜像标签，不使用 `latest` 漂移版本。

不要从旧教程照抄 `langfuse.callback` 路径或声称“编译时挂一次回调”必然覆盖当前全部调用；以后续 Context7 与实装版本试验为准。

## Task 1 implementation check (2026-09-27)

- Context7 `/langfuse/langfuse-python` confirms SDK v4 reads `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optional `LANGFUSE_BASE_URL`, with `CallbackHandler` in `langfuse.langchain`. The lock resolves `langfuse==4.15.6`.
- Context7 `/langfuse/langfuse-docs` and `/websites/langfuse_self-hosting` confirm the six-service Compose topology and approximate minimum resources. The exact [v4.46.0 upstream Compose](https://github.com/langfuse/langfuse/blob/v4.46.0/docker-compose.yml) was fetched before local edits; its web port 3000 and MinIO port 9090 were public bindings, now restricted to loopback. Upstream defaults such as `postgres`, `mysalt`, `miniosecret`, `myredissecret`, and a zero encryption key were removed in favor of required ignored local values.

## Task 2 implementation check (2026-09-27)

- Context7 `/langfuse/langfuse-python` and the installed 4.15.6 signatures show `CallbackHandler(*, public_key=None, trace_context=None)`; credentials and `base_url` belong to `Langfuse(...)`, whose client is retrieved by public key by the handler. The SDK `mask` hook covers SDK-owned input/output/metadata but is not a blanket mask for third-party OpenTelemetry spans.
- Context7 `/langfuse/langfuse-python` confirms `langfuse_session_id`, `langfuse_user_id`, and `langfuse_trace_name` in RunnableConfig metadata are interpreted as root trace attributes. Context7 `/langchain-ai/langgraph` confirms `ainvoke`/`astream` take RunnableConfig callbacks. A compiled Graph test exercised both paths with a failing callback and checked business results continue.

## Task 6 live SDK correction (2026-09-27)

- Context7 `/langfuse/langfuse-python` [MaskFunction protocol](https://github.com/langfuse/langfuse-python/blob/main/_autodocs/06-types-utilities.md) requires `def mask(*, data, **kwargs)`; installed SDK 4.15.6 called the old positional-only closure with `data=`, logged masking errors, and fell back. The live fake-model probe exposed this; a RED/GREEN test now uses the documented call form.
