# Ch01 Context7 MCP 核验记录

2026-09-23，使用已配置的 `https://mcp.context7.com/mcp`，通过 MCP `tools/call` 调用 `resolve-library-id` 与 `query-docs`。凭据不在此文件记录。

| 库 ID | 查询与采用结论 | 官方出处 |
|---|---|---|
| /websites/fastapi_tiangolo | StreamingResponse 接受 async generator；流中需要 await 让取消请求得到处理 | https://fastapi.tiangolo.com/advanced/custom-response |
| /websites/reference_langchain | trim_messages 支持 max_tokens、token_counter、strategy、start_on、end_on、allow_partial、include_system；可自定义估算器 | https://reference.langchain.com/python/langchain-core/messages/utils/trim_messages |
| /websites/reference_langchain | ChatOpenAI 提供 base_url、api_key、timeout、max_retries、use_responses_api；应用保持 OpenAI Chat Completions 协议 | https://reference.langchain.com/python/langchain-openai/chat_models/base |
| /websites/reference_langchain | with_structured_output 模式有 json_mode/json_schema/function_calling；本章排除 function_calling | https://reference.langchain.com/python/langchain-tests/unit_tests/chat_models/ChatModelTests |
| /websites/langchain_oss_python_langchain | OpenAI 兼容上游通过 base_url、api_key 配置；多轮消息使用 System/Human/AIMessage | https://docs.langchain.com/oss/python/langchain/models |
| /pydantic/pydantic-settings | SettingsConfigDict 配置 .env UTF-8，_env_file=None 可禁用 dotenv，初始化参数覆盖配置源 | https://github.com/pydantic/pydantic-settings/blob/main/docs/index.md |
| /astral-sh/uv | init --bare 建最简项目，add --dev 测试依赖，sync 按 lock 安装 | https://github.com/astral-sh/uv/blob/main/docs/concepts/projects/init.md |

安装锁定版本：FastAPI 0.141.1、LangChain 1.4.2、langchain-core 1.6.4、langchain-openai 1.6.3、pydantic-settings 2.15.0；完整解析结果以 uv.lock 为准。Context7 返回的引用可能跨集成模块，因此具体签名在使用前同时对照已安装源码，不能将其他供应商类的签名直接套给 ChatOpenAI。

2026-09-24 补充：Codex 原生 Context7 工具已加载，直接调用 query-docs 核验 FastAPI lifespan/TestClient 生命周期及 LangChain 估算器 chars_per_token 参数。继续使用已解析的官方库 ID；默认估算器对中文采用每字约 1 token 的较保守口径，但不宣称与所有供应商 tokenizer 精确一致。
