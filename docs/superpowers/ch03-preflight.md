# 第 3 章原始材料预检（待 brainstorm 裁定）

原始课程文件：`C:\Users\Administrator\Documents\PetShop_Helper开发文档\3\实战演练：动手跑通 RAG 链路.md`。已将其中 Spec 和 Plan 逐字拆存为 `specs/2026-09-25-ch03-rag-knowledge-base-source.md` 与 `plans/2026-09-25-ch03-rag-knowledge-base-source-plan.md`；它们是源材料快照，并非已经过当前项目评审的执行计划。

固定范围：BGE-M3、Milvus Lite、MySQL 知识原文权威源、pending → 向量 upsert → done 的可恢复双写，`query_faq(keyword)` 的入参出参契约不变；在线仅 dense Top-K。文档切分、历史对话挖 QA、`/kb` 录入页及三项浏览器验收均在源材料范围内。

待裁定的源文档矛盾：

1. Spec §3 指定 LangChain `OpenAIEmbeddings`，Plan 技术栈和 Task 2 用 OpenAI Python SDK。
2. Spec §4 同时写 `EMBED_API_KEY` 与 `SILICONFLOW_API_KEY`；`embed_model` 又同时写 `bge-m3` 与上游真实名 `BAAI/bge-m3`。Plan Task 1 明确真实名为 `BAAI/bge-m3`，但正式配置字段需统一。
3. 源 Plan 指定 LangChain 1.3，仓库已用 `langchain>=1.4.2`；须按当前依赖和 Context7 评审 API，不照抄旧示例。
4. Spec 预设从 `ch02-function-calling` 分支切出，仓库当前仅有 `feat/ch01-pure-chat`，第 2 章实际提交已在该分支。
5. Plan 的 `killpg` 为 POSIX 进程组处理方式，与当前 Windows 本机环境不匹配；作业停止机制需重新设计，同时保持作业管理需求。

Context7 初查：Milvus 官方文档示例为 `pymilvus[milvus-lite]` 加本地 `.db` URI；OpenAI 官方 Python SDK 支持 `AsyncOpenAI(base_url=...)` 的 embeddings 接口；LangChain Reference 有 `OpenAIEmbeddings`。Milvus Lite 新版[官方仓库说明](https://github.com/milvus-io/milvus-lite)已列 Windows 为有依赖 wheel 时可用的平台；旧版[官方 FAQ](https://milvus.io/docs/operational_faq.md)仍说不支持 Windows，二者反映版本差异。本机隔离环境已安装 `pymilvus==3.0.2`、`milvus-lite==3.2.1`，Windows/Python 3.12 实测本地 `.db` 建集合、insert、search、按 ID upsert、get 全部通过。原 Plan 按旧 2.x 结构写的 API 和返回值需要基于 Context7 与真实 3.x 行为复审。真实嵌入验收仍需独立 `EMBED_*` 凭据，现有聊天模型密钥不能假定适用。

当前已向用户询问嵌入客户端和配置字段名。问题定稿、计划评审及 Task 1 真实嵌入风险闸前，不改第三章运行代码、不调用付费嵌入上游。Milvus 冒烟只在 Git 忽略的 `work/` 隔离环境中运行，未修改项目依赖。

独立计划预审补充（当前源 Plan **未通过**，须产出修订执行计划并复审）：

- 源材料给出 `knowledge_chunks` 和 `qa_extraction_staging` DDL，但 Plan 没有创建 `sql/ch03-ddl.sql` 的任务；应在 ORM 和测试库之前补齐权威 DDL 及幂等应用步骤。
- 示例反复导入不存在的 `app.config.settings`，须适配现有 `get_settings()`。
- 本机没有 `make`；Plan 的页面作业全依赖 make，且 `show_kb.py`、`kb-preview`、`kb-reset` 尚无完整实现步骤。Windows 作业执行与停止机制需一并定稿。
- 挖知识先标 staging 为 kept 再写知识的步骤有崩溃漏入风险；源文档建库重跑可能重复插入。修订计划需稳定来源键、整体去重与失败恢复测试。
- 嵌入返回少于批量输入时，源代码 `zip` 会漏写向量却将整批标 done；标记 done 前必须校验数量与每条维度。
- Spec 要显式标记关键条款，Plan 却按关键词猜测；需定文档标记语法。挖知识 Prompt 的“忠于原答”与“统一改写时效”也冲突，需保持有依据的原答事实。
- Milvus Lite 是本地嵌入库，不能按源 Plan 用“关掉 Milvus 服务”测试故障；应在隔离环境注入读取失败。不得损坏正式数据文件。
