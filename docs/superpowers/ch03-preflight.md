# 第 3 章原始材料预检（待 brainstorm 裁定）

原始课程文件：`C:\Users\Administrator\Documents\PetShop_Helper开发文档\3\实战演练：动手跑通 RAG 链路.md`。已将其中 Spec 和 Plan 逐字拆存为 `specs/2026-09-25-ch03-rag-knowledge-base-source.md` 与 `plans/2026-09-25-ch03-rag-knowledge-base-source-plan.md`；它们是源材料快照，并非已经过当前项目评审的执行计划。

固定范围：BGE-M3、Milvus Lite、MySQL 知识原文权威源、pending → 向量 upsert → done 的可恢复双写，`query_faq(keyword)` 的入参出参契约不变；在线仅 dense Top-K。文档切分、历史对话挖 QA、`/kb` 录入页及三项浏览器验收均在源材料范围内。

待裁定的源文档矛盾：

1. Spec §3 指定 LangChain `OpenAIEmbeddings`，Plan 技术栈和 Task 2 用 OpenAI Python SDK。
2. Spec §4 同时写 `EMBED_API_KEY` 与 `SILICONFLOW_API_KEY`；`embed_model` 又同时写 `bge-m3` 与上游真实名 `BAAI/bge-m3`。Plan Task 1 明确真实名为 `BAAI/bge-m3`，但正式配置字段需统一。
3. 源 Plan 指定 LangChain 1.3，仓库已用 `langchain>=1.4.2`；须按当前依赖和 Context7 评审 API，不照抄旧示例。
4. Spec 预设从 `ch02-function-calling` 分支切出，仓库当前仅有 `feat/ch01-pure-chat`，第 2 章实际提交已在该分支。
5. Plan 的 `killpg` 为 POSIX 进程组处理方式，与当前 Windows 本机环境不匹配；作业停止机制需重新设计，同时保持作业管理需求。

Context7 初查：Milvus 官方文档示例为 `pymilvus[milvus-lite]` 加本地 `.db` URI；OpenAI 官方 Python SDK支持 `AsyncOpenAI(base_url=...)` 的 embeddings 接口；LangChain Reference 有 `OpenAIEmbeddings`。资料尚未证明 Milvus Lite 在本机 Windows 可用，须先做安装/启动冒烟，不能未经用户同意换向量库。真实嵌入验收还需独立 `EMBED_*` 凭据，现有聊天模型密钥不能假定适用。

当前已向用户询问嵌入客户端和配置字段名。问题定稿、计划评审及 Task 1 风险闸前，不改第三章运行代码、不调用付费嵌入上游。
