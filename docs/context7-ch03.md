# 第 3 章 Context7 接口核对

2026-09-25 在实现前经 Context7 查询官方资料：

- **Milvus / PyMilvus**：本地 `.db` URI 由 `MilvusClient` 打开，集合可用 `create_collection`、`insert`、`search`、`upsert`、`get`。参考 [Milvus Lite 官方文档](https://milvus.io/docs/milvus_lite.md)及[Milvus Lite 新版仓库](https://github.com/milvus-io/milvus-lite)。旧 FAQ 称 Windows 不支持，但新版仓库注明依赖 wheel 可用；本机 Windows/Python 3.12 用 `pymilvus==3.0.2`、`milvus-lite==3.2.1` 完成上述 API 冒烟。
- **OpenAI Python SDK**：`AsyncOpenAI(api_key=..., base_url=...)` 可对 OpenAI 兼容端点调用 `await client.embeddings.create(model=..., input=[...])`，返回每条 `embedding`。参考 [官方 SDK README](https://github.com/openai/openai-python)和[API 文档](https://github.com/openai/openai-python/blob/main/api.md)。设计文档已定用此 SDK，而非前文旧版 `OpenAIEmbeddings` 写法。
- **Pydantic Settings**：`BaseSettings` 从 `.env` 与环境变量加载同名字段，`SecretStr` 隐藏密钥表示，字段约束使用 `Field`；如需不同名称可使用 `validation_alias`。参考 [官方 Settings 文档](https://github.com/pydantic/pydantic-settings/blob/main/docs/index.md)。项目正式字段直接命名 `siliconflow_api_key`，映射 `SILICONFLOW_API_KEY`。

后续使用 LangChain 文本切分器、SQLAlchemy 新表映射、FastAPI `/kb` API 时，应在各任务动手前分别核对最新官方接口并追加记录。

## Task 6：LangChain 文本切分器（2026-09-25）

经 Context7 `/langchain-ai/docs` 核对 [Markdown 标题切分官方示例](https://github.com/langchain-ai/docs/blob/main/src/oss/python/integrations/splitters/markdown_header_metadata_splitter.mdx)及[递归切分官方示例](https://github.com/langchain-ai/docs/blob/main/src/oss/integrations/splitters/recursive_text_splitter.mdx)：`MarkdownHeaderTextSplitter(headers_to_split_on=..., strip_headers=True).split_text(md)` 返回带标题 metadata 的 `Document` 列表；`RecursiveCharacterTextSplitter(chunk_size=..., chunk_overlap=..., separators=..., is_separator_regex=False, length_function=len).split_text(text)` 返回字符串块。当前 `langchain>=1.4.2` 未附带可导入的 `langchain_text_splitters`，已显式增加 `langchain-text-splitters==1.1.2`（锁文件解析结果）。
