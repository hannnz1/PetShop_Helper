# Ch03 执行裁定（源 Spec/Plan 的约束补充）

源文档：[Spec](../specs/2026-09-25-ch03-rag-knowledge-base-source.md)、[Plan](2026-09-25-ch03-rag-knowledge-base-source-plan.md)。本文件以原设计 §14 决策记录和 Plan Global Constraints 为准，修复独立预审的执行缺口；固定 BGE-M3、Milvus Lite、MySQL 权威源、dense 单路及 `query_faq` 接口契约。

1. **嵌入配置按设计后段决策。** 使用 OpenAI Python SDK 的 `AsyncOpenAI` 调 SiliconFlow OpenAI 兼容 `/v1/embeddings`；正式密钥字段 `SILICONFLOW_API_KEY`，模型 ID `BAAI/bge-m3`，地址从配置读取，默认 `https://api.siliconflow.cn/v1`。Spec §3 的 `OpenAIEmbeddings`、Spec §4 / Plan Task 1 的 `EMBED_API_KEY` 是前段未同步写法。配置字段可统一称 `embed_api_key`，但从环境映射 `SILICONFLOW_API_KEY`，不从聊天密钥自动回退。缺密钥时只允许离线 Milvus/切块等测试，不标真实嵌入 GO。
2. **Task 0 加权威 DDL。** 在原 Task 1 前，从用户课程文档的 SQL 块完整整理 `sql/ch03-ddl.sql`，含 `knowledge_chunks` 和 `qa_extraction_staging`、`SET NAMES utf8mb4`。新建数据库表仅执行新增 DDL，不清理业务库；测试 fixture 只在隔离 `*_test` 库执行该 DDL。先有 DDL，再做 ORM/仓储任务。
3. **保持现有配置接口。** 所有示例 `from app.config import settings` 改为 `get_settings()` 或传入 `Settings`；现有 `langchain>=1.4.2` 不降回旧 Plan 的 1.3，具体 splitter/SQLAlchemy/FastAPI/Milvus API 在实现前逐项用 Context7 查官方文档。
4. **Milvus Lite 按本机证实版本实施。** Windows/Python 3.12 隔离冒烟已用 `pymilvus==3.0.2`、`milvus-lite==3.2.1` 完成本地 `.db` 的 create/insert/search/upsert/get。正式依赖应包含两个包并锁定兼容版本范围；不得照抄源 Plan 的 2.x 返回值假设。正式集合使用 `knowledge`、MySQL id 作向量主键，维度须等真实 BGE-M3 冒烟确认为 1024 后才能创建。
5. **严格真实嵌入闸。** Task 1 真实请求输入两句中文，只报告返回向量数量和维度，不输出密钥、请求头或全量向量；必须两条均为 1024 维才 GO。当前 `.env` 没有 `SILICONFLOW_API_KEY`，此闸目前未过；不以聊天密钥、fake 或本地 Milvus 冒烟冒充。真实闸失败时按用户要求停问，不更换模型或供应商。
6. **双写不能假成功。** 批量嵌入返回数量须等于输入数量且每条维度匹配；否则当前批保持 pending。Milvus upsert 成功后逐 id 回填 done；任何一处失败不将未确认块标 done。测试注入少返回、维度错误、upsert 后 DB 写失败和批中断，重跑后 MySQL done 数及 Milvus id 集一致。
7. **离线两条写入链幂等。** 源文档按来源路径、章节位置及“问法 + 正文”归一化指纹避免重灌；手工录入按设计 §10.2 同一指纹去重，且同章节不同正文保留。源 DDL 没有指纹/来源唯一列，不能未经评审新增字段；在现有列上完成内容比对，并以事务及单作业互斥保证写入，同时测试并发。同一用户改正文的行为须在计划评审明确，不在本章默默覆盖。历史会话分批使用稳定 `batch_no`/真实 `source_ref`；抽取先落 staging，去重后将知识写入与 staging kept 状态在同一 MySQL 事务确认，或提供能扫描 kept 未入库的恢复步骤。必须测“中断后重跑无重复无漏”。
8. **关键条款按显式标记。** 源 Markdown 用明确的 `<!-- key-clause -->` 标记其后一个段落/表格块；切块器移除标记并设置该块 `is_key_clause`。不能仅凭“时效”“退款”等词猜测条款级别。源夹具、预览和入库走同一实现，含标记和无标记样例。
9. **挖知识 Prompt 保真。** 从真实客服回答提炼通用、可复用事实；不能将原文具体时效统一改成另一句话，也不能把未经核实的承诺写入知识库。纯 Prompt 用标注样例评估，覆盖保留原答依据、排除具体订单隐私、拒绝编造和跨批不串味。
10. **Windows 页面作业仍走 make。** 本机 GNU Make 在 `C:\msys64\usr\bin\make.exe`；Windows 作业可显式定位该路径，Unix 用 PATH，最终都执行同一 Makefile 的白名单目标，shell 参数不接受前端输入。`make -n seed` 只验证旧目标可解析；新增 kb 目标须用 Makefile 平台分支或平台中性 Python CLI，不照抄 POSIX `PYTHONPATH=. uv run ...`，并实际运行 Windows 配方。补齐 `show_kb.py`、`kb-preview`、`kb-reset`、`seed-conv`、`eval-mining` 等后续页面/验收实际引用的脚本或目标，并逐一验证。停止作业在 Windows 用受控进程树终止机制，不能使用 POSIX `killpg`；启动、轮询、停止需有平台测试。
11. **录入页范围不缩减。** `/kb` 的材料清单/切块预览、手工录入与查重、补 pending、检索自测和作业日志，以及 `/admin` 聚合入口均是源 Spec 范围。聊天页仅消费不变的 `query_faq` 契约；此章不加入关键词、混合检索或重排。页面视觉交互遵照用户的 Vibe Coding 例外，后端 API/作业走 TDD。
12. **验收故障注入不损坏正式库。** Milvus Lite 是进程内文件库，不能按源 Plan “关掉 Milvus 服务”；读不到状态在隔离测试中以不可用 URI/注入连接错误验证。正式数据文件不得故意损坏。最终验收同时核对“邮费是多少”召回运费规则、崩溃后 pending 补齐、浏览器全链操作与标注集，不将单一工具选择通过率当作回答质量。

原 19 项任务按上述裁定调整执行顺序：先补 Task 0 DDL，再完成 Task 1 真实嵌入闸；Task 2–19 保持分任务 TDD/Prompt eval/独立评审。真实闸未过时只执行不依赖嵌入的离线准备，不标记后续任务完成。
