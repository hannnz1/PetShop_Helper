# 第 9 章第一部分：本地可观测性与评估趋势 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让每轮客服对话可在本机 Langfuse 定位调用链，并形成按意图的真实 token 账与可比较的 RAG 评估趋势。

**Architecture:** Langfuse 保存调用链明细，Graph 每轮运行配置附加回调；应用 MySQL 只保存脱离正文的模型用量事件和评估运行摘要。评估包装器复用第 4 章既有指标，缺外部服务时标记不可用。独立 Docker Compose 运行 Langfuse，不干扰当前容器与卷。

**Tech Stack:** Python 3.12、FastAPI、LangGraph、LangChain、Langfuse Python SDK 4.x、Docker Compose、SQLAlchemy/MySQL。

**Spec:** `docs/superpowers/specs/2026-09-27-ch09-observability-evaluation-design.md`

## Global Constraints

- 用户选择本机 Docker 自托管 Langfuse；没有授权 Cloud trace 外发。Docker Compose 使用官方固定版本及独立数据卷，真实密钥仅在 Git 忽略的本地环境文件。
- 不改变第 8 章 Graph、MCP、确认建单的业务行为；观测关闭或失联不得中断 SSE/JSON 对话。
- 统计表只保留会话/本轮关联号、意图、模型、真实 token 数或 unavailable 状态，不保留正文及密钥；不基于字符数伪造 token。
- 第 4 章评估指标和样例集定义不变；`pending_upstream` 不是通过，不重算或覆盖历史分数。
- 每个代码任务先 RED 后 GREEN 并在 `dev-notes/ch09.md` 即时追记，完成时独立后端 code review。页面若需修改按用户要求走 Vibe Coding 例外。
- 每次涉及具体库 API 先查 Context7 官方资料并记入 `docs/context7-ch09.md`；实装版本与文档不符时停下核查。

## Review Focus

1. 观测启用但本地 Langfuse 不可达：`tests/observability/test_tracing.py::test_callback_failure_does_not_break_graph_turn` 证明业务继续。
2. 外部 Langfuse URL 或缺一把密钥：`tests/observability/test_tracing.py::test_invalid_host_or_partial_credentials_disables_tracing` 证明不外发、不半配置启动。
3. 流式响应末尾重复 usage/重试重复 run ID：`tests/observability/test_usage.py::test_same_run_id_is_counted_once` 证明无双算。
4. 模型不返回 usage：`tests/observability/test_usage.py::test_missing_usage_is_unavailable` 证明报表不记假零。
5. 不同数据集 hash 或分母的评估记录：`tests/observability/test_eval_trend.py::test_incomparable_runs_are_separated` 证明趋势不误连。

---

### Task 1: 本地 Langfuse 配置与可用性

**Files:**
- Create: `infra/langfuse/compose.yaml`, `infra/langfuse/.env.example`, `scripts/ch09/langfuse_local.ps1`, `docs/ch09-local-langfuse.md`
- Modify: `.gitignore`, `.env.example`, `app/config.py`, `pyproject.toml`, `uv.lock`, `dev-notes/ch09.md`
- Test: `tests/observability/test_config.py`

**Interfaces:**
- Produces: `Settings.langfuse_enabled: bool = False`, `langfuse_base_url`, `langfuse_public_key`, `langfuse_secret_key`, `observability_admin_token: SecretStr | None`；部署脚本 `-Action Start|Stop|Status`。

- [ ] **Step 1:** 用 Context7 查当前 Langfuse Python SDK 4.x 配置和官方 v4.46.0 Compose，再核对被下载/保存的镜像标签、端口及默认口令。
- [ ] **Step 2:** 写 `test_langfuse_defaults_off_and_keys_hidden`，断言关闭状态无需密钥，`SecretStr` 不泄露；先运行观察 RED。
- [ ] **Step 3:** 增加设置、依赖与本地 Compose。将官方固定版本 Compose 的开放端口限制到 `127.0.0.1`，所有服务使用独立命名卷，真实 `.env.langfuse` 被忽略；脚本启动前检查 Docker 资源/端口且不调用 `down -v`。
- [ ] **Step 4:** 运行配置测试、`docker compose --env-file <本地文件> -f infra/langfuse/compose.yaml config`、`uv pip check`；`docs/ch09-local-langfuse.md` 写出安全初始化、密钥配置和停机命令。无本地凭据时先验证配置和关闭路径，真实启动留最终集成任务。
- [ ] **Step 5:** 在 `dev-notes/ch09.md` 追记关键原话、产出/测试、纠偏、翻车返工，提交 Task 1。

### Task 2: Graph 每轮 trace 与故障隔离

**Files:**
- Create: `app/observability/tracing.py`, `tests/observability/test_tracing.py`
- Modify: `app/graph/runtime.py`, `app/main.py`, `dev-notes/ch09.md`

**Interfaces:**
- Consumes: Task 1 的 Langfuse 设置。
- Produces: `make_turn_callbacks(settings, user_id: str, conversation_id: int, turn_id: str) -> list[BaseCallbackHandler]`，在 `ainvoke_turn`、`prepare_stream_turn`、`prepare_resume_turn` 的 RunnableConfig 附加；`turn_id` 供 Task 3 用量关联。

- [ ] **Step 1:** 通过 Context7 确认 SDK 4.x `CallbackHandler`、RunnableConfig metadata 与当前 LangGraph 流式传播接口；固定匿名 user 标签和会话/本轮 ID 映射。
- [ ] **Step 2:** 写 `test_invoke_stream_and_resume_attach_one_trace`、Review Focus 1/2 的失败测试；验证关闭时零回调、开启时三入口有一致关联字段、Cloud URL 被拒、回调异常不改变业务结果。运行 RED。
- [ ] **Step 3:** 实现按轮回调工厂和 Graph 三入口接线；只在本机/明确自托管内网地址及双密钥齐备时初始化，SDK 失败记录应用日志。不要把 API key/DSN 放入 metadata。
- [ ] **Step 4:** 运行新测试和现有 `tests/graph tests/test_chat_api.py tests/test_agent_api.py`，确认 SSE 完成帧与 Graph 中断恢复不回归。
- [ ] **Step 5:** 追记 `dev-notes/ch09.md` 并提交 Task 2。

### Task 3: 实际模型用量事件与意图结算

**Files:**
- Create: `app/observability/usage.py`, `app/db/observability.py`, `sql/ch09-observability.sql`, `tests/observability/test_usage.py`
- Modify: `app/db/models.py`, `tests/conftest.py`, `app/graph/runtime.py`, `app/core/llm.py`, `app/config.py`, `.env.example`, `dev-notes/ch09.md`

**Interfaces:**
- Produces: `UsageCollector(turn_id: str, conversation_id: int)` 收集 `on_llm_end` 的真实 `AIMessage.usage_metadata`，`finalize(intent: str) -> list[UsageEvent]`；`save_usage_events(events: Sequence[UsageEvent]) -> None` 按模型 run ID 幂等写入。Task 4 从同一表汇总。

- [ ] **Step 1:** Context7 核对当前 LangChain `on_llm_end` / `AIMessage.usage_metadata` 和 ChatOpenAI `stream_usage`；用最小假模型探针确认流式末尾只结算一次。
- [ ] **Step 2:** 写 `test_same_run_id_is_counted_once`、`test_missing_usage_is_unavailable`、`test_final_intent_applies_to_early_model_calls` 与 MySQL 持久化测试；先运行 RED。
- [ ] **Step 3:** 实现仅存数字与关联 ID 的 ORM/DDL、buffered collector 和结算；正常完成从 Graph 最终 intent 结算，失败/中断用 `unknown` 且状态明确；审计重复回调/重试不双算。现有 `stream_usage=False` 保持默认，以 `CHAT_STREAM_USAGE` 显式开启支持该协议的上游，用量不可得时显示 unavailable；开启后不得改变逐 token SSE。
- [ ] **Step 4:** 在隔离 `mewhelp_test` 运行新测试、Graph/Chat 定向回归，检查数据库没有 prompt、回复或密钥列。
- [ ] **Step 5:** 追记 `dev-notes/ch09.md` 并提交 Task 3。

### Task 4: 意图成本报表

**Files:**
- Create: `app/api/observability.py`, `scripts/ch09/usage_report.py`, `tests/observability/test_usage_report.py`
- Modify: `app/db/observability.py`, `app/main.py`, `Makefile`, `dev-notes/ch09.md`

**Interfaces:**
- Consumes: Task 3 用量事件表。
- Produces: `usage_by_day(start: date, end: date) -> list[dict]`；只读 `GET /api/observability/usage?from=YYYY-MM-DD&to=YYYY-MM-DD` 和 `make usage-ch09`。

- [ ] **Step 1:** 写两天、两意图、两模型和 unavailable 混合数据测试，断言输入/输出 token 与调用次数汇总正确、无货币金额、非法日期及无 Bearer token 被拒绝；运行 RED。
- [ ] **Step 2:** 实现 MySQL 聚合、只读 API 与 CLI JSON/表格输出；现有 `/api/admin` 没有鉴权，因此本 API 必须校验 `OBSERVABILITY_ADMIN_TOKEN` 的 Bearer 值，未配置时拒绝 HTTP 访问，不能将用量公开给未授权浏览器。
- [ ] **Step 3:** 运行 `tests/observability/test_usage_report.py` 及相关 API 回归并人工核对演示报表。
- [ ] **Step 4:** 追记 `dev-notes/ch09.md` 并提交 Task 4。

### Task 5: 第 4 章评估结果入账与趋势

**Files:**
- Create: `scripts/ch09/eval_trend.py`, `app/observability/eval_trend.py`, `tests/observability/test_eval_trend.py`, `docs/ch09-eval-schedule.md`
- Modify: `app/db/models.py`, `app/db/observability.py`, `sql/ch09-observability.sql`, `app/api/observability.py`, `Makefile`, `dev-notes/ch09.md`

**Interfaces:**
- Produces: `record_eval_run(run_id: str, dataset_hash: str, strategy: str, report: dict) -> None` 幂等记录摘要；`comparable_trend(dataset_hash: str, strategy: str) -> list[dict]`；`make eval-ch09` 一次运行并退出。

- [ ] **Step 1:** 为纯数据提取准备第 4 章报告标注样例；写缺指标、两个不同 hash、重复 run ID、分母变化和 `pending_upstream` 样例，先跑一遍确认提取/比较规则失败（纯数据逻辑用标注样例代替调用付费模型）。
- [ ] **Step 2:** 实现摘要表/DDL、报告解析器与趋势查询；真实包装命令调用既有 `scripts.eval_ch04`，每次保存本地原报告和 Git SHA，不覆盖历史。缺外部服务时只记不可用，不填旧分数。
- [ ] **Step 3:** 跑样例评估、隔离 MySQL 测试、CLI 两次运行与趋势 API；文档给 Windows 任务计划程序的单次命令，不自动在 Web 请求里调模型。
- [ ] **Step 4:** 追记 `dev-notes/ch09.md` 并提交 Task 5。

### Task 6: 本机集成验收、独立复核与交付

**Files:**
- Create: `docs/ch09-observability-acceptance.md`
- Modify: `docs/ch09-local-langfuse.md`, `dev-notes/ch09.md`

**Interfaces:**
- Consumes: Tasks 1–5 全部产物。
- Produces: 可复现的部署/演示命令、测试结果与未满足前提清单。

- [ ] **Step 1:** 启动独立 Langfuse Compose，确认健康、建本地项目与密钥；用假模型在隔离 app/test 库生成真实 trace，核对本机界面的 trace 树、节点关联与无密钥泄漏。若资源不足，记录具体错误并继续不依赖 Langfuse 的测试，不换 Cloud。
- [ ] **Step 2:** 停止 Langfuse 重放一轮 Graph，验证客服继续；验证 intent token 报表和评估趋势的真实 API/CLI 路径。恢复原环境并保留数据卷。
- [ ] **Step 3:** 跑第 9 章定向测试及可用环境下广域回归；真实 glm-5.2 / Milvus / SiliconFlow 的指标有依赖时才运行，额度不足记 `pending_upstream`。每项验收写证据与限制。
- [ ] **Step 4:** 使用 `superpowers:requesting-code-review` 进行独立后端 review，按 `receiving-code-review` 修复重要问题后复测。
- [ ] **Step 5:** 在 `dev-notes/ch09.md` 即时追记 review 结论、返工与 finish 四项，提交报告和修复。交付演示命令、测试结果、spec/plan/dev-notes 路径；第 9 章第二部分另走书面设计和计划评审。

## Self-Review

- Spec 的部署、trace、usage、评估和四条验收分别落在 Task 1、2、3–4、5、6；低置信度回流明确为下一子项目。
- 各任务接口统一使用 `turn_id` / `run_id`，模型用量 run ID 与评估 run ID 分属不同表；缺 usage/缺依赖都用明确状态，不以 0 代替。
- 每个 Review Focus 情况都有本任务测试。非代码 Compose 先用配置验证；标注数据提取用样例验证，其余功能 TDD。
- 任务按可独立复核的组件切分；后续实现若实装库 API 与计划冲突，先以 Context7 和版本探针核对，不擅自换技术栈。
