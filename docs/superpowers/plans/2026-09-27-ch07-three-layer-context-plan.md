# Chapter 7 Three-Layer Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让单通客服会话保留完整可回载原文，并在模型窗口内按近期原文、中期规则缩短、早期异步只追加摘要提供上下文。

**Architecture:** SQLite checkpoint 继续保留 LangGraph 全消息；MySQL 原文消息的两个 ID 锚点决定输入层次，独立分段摘要表保存第三层。独立预算器与上下文组装器只生成模型视图；SSE 成功后调度摘要，页面通过本人只读 API 切换会话。

**Tech Stack:** Python 3.12、FastAPI、LangGraph、LangChain、SQLAlchemy 2.0/MySQL、SQLite checkpoint、原生 HTML/CSS/JS。

**Spec:** [第 7 章已批准设计](../specs/2026-09-27-ch07-three-layer-context-design.md)。具体 API 先核对 [Context7 记录](../../context7-ch07.md)。

## Global Constraints

- 仅当前会话，不做跨会话记忆、用户画像或语义检索历史；保留第 5/6 章图和退款确认语义。
- 不调用余额不足的付费 glm 上游；真实二十轮和真实摘要验收标 `pending_upstream`，离线假模型结果单列。
- `summary_upto_msg_id`、`layer1_from_msg_id` 单调前进；摘要段只追加，旧段不重写；原文不删改。
- `messages` 中新图审计只写用户/客服原文，旧工具行不删除；Graph checkpoint 继续保存完整消息和工具交互。
- 预算按模型窗口倒推；演示配置 `MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5`，目标历史约 5650（第一层 3954、第二层 1695）。
- 上下文顺序固定，摘要+证据在当前用户消息之后；记录 `model_ctx`、`history_ctx` 和摘要任务事件；日志含原文，应仅本机可读。
- 确定性后端先红后绿；摘要 Prompt 用标注样例验证；聊天页面按用户指定的 Vibe Coding 直接迭代。
- 每完成一项立即追记 `dev-notes/ch07.md` 四项信息并提交。写新库接口前先补查 Context7。

## File Map

- `sql/ch07-ddl.sql`、`scripts/migrate_ch07.py`、`app/db/models.py`、`app/db/repository.py`：非破坏性迁移、分段摘要和有归属的会话读取。
- `app/core/context_budget.py`：配置派生与预算校验；`app/core/context_layers.py`：完整轮次分层、规则缩短和模型消息拼装。
- `app/core/summarizer.py`、`app/core/prompts.py`：摘要选择、后台生命周期和只提事实 Prompt。
- `app/config.py`、`app/core/memory.py`、`app/graph/state.py`、`app/graph/nodes.py`、`app/graph/runtime.py`、`app/api/chat.py`：将精简视图接入 Graph、日志、SSE 完成后的调度。
- `app/api/conversations.py`、`app/main.py`：本人会话列表与原文分页接口；`app/static/index.html`：侧栏会话切换。
- `tests/test_ch07_budget.py`、`tests/test_ch07_layers.py`、`tests/test_ch07_repository.py`、`tests/test_ch07_summarizer.py`、`tests/graph/test_ch07_context.py`、`tests/test_ch07_conversations_api.py`：定向回归。
- `tests/data/ch07_summary_cases.jsonl`、`scripts/eval_ch07.py`、`data/ch07/reports/offline_eval.json`：标注案例与离线报告；`README.md`、`dev-notes/ch07.md`：演示与逐阶段记录。

## Review Focus

1. 旧 `messages` 中有 `role=tool` 和不连续全局 ID → 不能进入可见侧栏或造成锚点跨会话；Task 1/3/6 测。
2. 摘要任务与下一轮同时推进锚点 → 同一区间最多一段，失败不跳过历史；Task 5 测。
3. SSE 提前断开、订单选择 pending、MySQL 审计失败 → 不发摘要任务；Task 4/5 测。
4. 系统提示、检索证据或某一轮工具结果过大 → 明确预算错误，不丢用户原话或证据；Task 2/3/4 测。
5. 切换会话时旧请求回包或旧 pending 表单到达 → 不覆盖当前会话，也不误提交订单/退款；Task 7 验。

---

### Task 1: 非破坏性 Schema 与完整原文边界

**Files:** Create `sql/ch07-ddl.sql`, `scripts/migrate_ch07.py`, `tests/test_ch07_repository.py`; modify `app/db/models.py`, `app/db/repository.py`, `tests/graph/test_audit_recovery.py`, `dev-notes/ch07.md`.

**Interfaces:** `ContextSnapshot` is a dataclass in `app/db/repository.py` with conversation ID, two anchors, ordered visible messages after the summary anchor, and immutable summary segments. `get_context_snapshot(conversation_id: int, user_id: str) -> ContextSnapshot | None`; `advance_layer1(conversation_id: int, upto_msg_id: int) -> bool` only advances to a completed owned turn; `append_turn_messages` returns final assistant ID and no longer inserts tool rows.

- [ ] 写 MySQL 失败测试：迁移二次执行不清库；原文中旧工具行仍存在但快照过滤；两个会话 ID 交错不串边界；新增图轮只写 `user,assistant`；旧 checkpoint marker 仍等于最后客服行。运行 `pytest -q tests/test_ch07_repository.py tests/graph/test_audit_recovery.py`，确认新断言红。
- [ ] 实现 SQL/ORM/仓储及迁移脚本；旧 `summary` 为空时两锚点视为初始值；分段 `(conversation_id, seq)` 唯一且 `from<=upto`。
- [ ] 在隔离库跑迁移两次和定向测试，确认通过；提交代码、测试及本任务 `dev-notes/ch07.md` 记录。

### Task 2: 预算器与中文 token 校准

**Files:** Create `app/core/context_budget.py`, `tests/test_ch07_budget.py`; modify `app/config.py`, `app/core/memory.py`, `.env.example`, `dev-notes/ch07.md`.

**Interfaces:** `FixedCosts` and `ContextBudget` are dataclasses in `app/core/context_budget.py`; `derive_budget(settings: Settings, fixed: FixedCosts) -> ContextBudget` returns `history_total`, `layer1`, `layer2`, `current_peak`; `validate_context_budget(settings: Settings, fixed: FixedCosts) -> None` raises `ContextBudgetExceeded` when one turn cannot fit. `estimate_tokens` retains a single configured CJK calibration point.

- [ ] 写失败测试：演示设置导出约 5650/3954/1695；默认二十个短轮次可留在第一层；只缩窗口会在启动自检失败；系统/证据/摘要/输出/当前 ReAct 峰值均计入；中文校准变动同步影响层预算和估算。运行 `pytest -q tests/test_ch07_budget.py`，确认红。
- [ ] 实现配置字段、固定开销测量、历史目标与 70/30 分配；显式处理旧 `TOKEN_BUDGET` 兼容，不绕过自检。使用 Context7 确认的 LangChain 估算 API。
- [ ] 跑定向和 `tests/test_memory.py`；提交并追记测试数据与偏差。

### Task 3: 三层渲染与模型输入顺序

**Files:** Create `app/core/context_layers.py`, `tests/test_ch07_layers.py`; modify `app/graph/state.py`, `dev-notes/ch07.md`.

**Interfaces:** `ModelContext` is a dataclass in `app/core/context_layers.py` with ordered `messages`, injected summary, token count and window rows. `build_model_context(snapshot: ContextSnapshot, graph_messages: list[BaseMessage], current_query: str, evidence: str, system_text: str, budget: ContextBudget) -> ModelContext`; `build_history_context(snapshot: ContextSnapshot, budget: ContextBudget) -> str` returns summary plus trimmed prior visible turns. `trim_messages` is final whole-turn gate.

- [ ] 写失败测试：两锚点按全局消息 ID 但仅本会话筛选；第一层用户/客服逐字保留，第二层用户不变/客服缩短，checkpoint 大工具内容一行标识；第三层旧段只读；最后一条摘要+证据位于当前问题之后；过大当前轮抛预算错误；无半轮、无跨会话混入。运行 `pytest -q tests/test_ch07_layers.py`，确认红。
- [ ] 实现纯组装器，不改 checkpoint `messages`；复用 LangChain `trim_messages` 和统一 token 估算；模型视图与后台摘要输入保持不同对象。
- [ ] 跑定向与 `tests/test_memory.py`；提交并追记。

### Task 4: Graph 接入与每轮上下文日志

**Files:** Create `tests/graph/test_ch07_context.py`; modify `app/graph/nodes.py`, `app/graph/runtime.py`, `app/api/chat.py`, `app/main.py`, `dev-notes/ch07.md`.

**Interfaces:** Graph 入口按会话 ID 读取 `ContextSnapshot` 并提供给节点；`coref` 和分类共用 `build_history_context`；每次 `agent_llm`/`final_answer` 调用 `build_model_context` 并写 `model_ctx`；每轮 `coref` 前写 `history_ctx`，即使 route 后续兜底。`GraphRuntime` 仍在审计分歧时拒绝续聊。

- [ ] 写失败测试：连续两轮指代接住早期摘要；工具循环的每步实际上下文重算；闲聊兜底也打 `history_ctx`；`model_ctx` 逐条匹配实际传给假模型的消息；订单中断/恢复及 audit marker 不回退；流失败没有完成或摘要调度。运行 `pytest -q tests/graph/test_ch07_context.py tests/graph/test_ch06_resume.py`，确认红因新断言。
- [ ] 接入组装器和日志，保留现有 SSE 帧/退款权限；异常预算返回明确错误；成功完成帧只表示原文与 checkpoint 成功，不等待后台摘要。
- [ ] 跑定向、`tests/test_ch05_chat_stream.py` 和第 6 章回归；提交并追记。

### Task 5: 异步分段摘要及失败恢复

**Files:** Create `app/core/summarizer.py`, `tests/test_ch07_summarizer.py`, `tests/data/ch07_summary_cases.jsonl`; modify `app/core/prompts.py`, `app/db/repository.py`, `app/graph/runtime.py`, `app/api/chat.py`, `app/main.py`, `docs/context7-ch07.md`, `dev-notes/ch07.md`.

**Interfaces:** `SummaryOutcome` is a dataclass in `app/core/summarizer.py` with `status: Literal["skipped", "running", "committed", "failed"]` and covered IDs. `schedule_summary(conversation_id: int) -> None` enqueues only after a completed SSE turn; `summarize_pending(conversation_id: int, model: BaseChatModel) -> SummaryOutcome` checks Layer2 token use, summarizes one contiguous completed range, appends `ConversationSummary`, advances `summary_upto_msg_id` atomically without changing response status.

- [ ] 写失败测试：阈值按 token 不按条数；摘要调用在 `[DONE]` 后才开始且不阻塞该帧；并发两任务只提交一段；模型失败边界不动且下轮重试；无事实写跳过标记避免死循环；旧段内容永不回炉；流断开/pending 不调度。运行 `pytest -q tests/test_ch07_summarizer.py`，确认红。
- [ ] 实现只提事实的 Prompt、后台生命周期、数据库原子提交与 `trigger/start/done/skip/fail` 日志；进程退出有界等待，重启凭锚点补查。纯 Prompt 质量用标注样例逐例评估，不用镜像实现的单测替代。
- [ ] 跑定向和标注样例离线验证，提交并追记摘要质量、失败路径。

### Task 6: 本人只读会话接口

**Files:** Create `app/api/conversations.py`, `tests/test_ch07_conversations_api.py`; modify `app/db/repository.py`, `app/main.py`, `dev-notes/ch07.md`.

**Interfaces:** `GET /api/conversations?user_id=<id>&limit=<n>&before=<cursor>` returns newest-first previews, summary flag and next cursor; `GET /api/conversations/{id}/messages?user_id=<id>&limit=<n>&after=<msg_id>` returns visible original messages ascending and next cursor. Missing/foreign conversation is 404.

- [ ] 写失败测试：本人多会话排序/分页/首问预览；他人列表不泄漏；他人或不存在消息 404；旧工具行不回传；多页可完整回载；非法页大小 422。运行 `pytest -q tests/test_ch07_conversations_api.py`，确认红。
- [ ] 实现有上限的查询和 FastAPI 路由，维持当前演示 `user_id` 身份边界，不扩大到认证项目。
- [ ] 跑定向与聊天 API 回归；提交并追记。

### Task 7: 聊天页多会话侧栏（Vibe Coding 例外）

**Files:** Modify `app/static/index.html`, `dev-notes/ch07.md`.

**Interfaces:** 页面调用 Task 6 的两个 GET；`newChat()` 保留旧会话可切回；pending 订单按会话 ID 恢复；侧栏失败不妨碍现有发送。

- [ ] 直接做侧栏样式和交互：最近会话、首问预览、摘要标记、选中态、逐页历史回载；对齐现有桌面/移动布局。
- [ ] 用本地浏览器或等效 UI 操作验证两个会话切换后各自原文完整、旧 SSE 回包不会覆盖当前会话、旧 pending 订单不会显示在新会话、侧栏 API 失败仍能新聊；按看到的效果迭代。
- [ ] 提交页面并立即追记。页面按用户授权不做独立 brainstorm/TDD/code review，后端契约仍由 Task 6 测试覆盖。

### Task 8: 离线验收、回归、文档与独立复核

**Files:** Create `scripts/eval_ch07.py`, `data/ch07/reports/offline_eval.json`; modify `README.md`, `dev-notes/ch07.md`; run existing `tests/`.

**Interfaces:** `python scripts/eval_ch07.py --offline` writes separate pass/fail/pending-upstream counts and asserts marked cases from `tests/data/ch07_summary_cases.jsonl`.

- [ ] 先写标注样例与失败的评估器断言：默认 20 轮无压缩；演示 20+ 轮级联、最早订单事实可由摘要召回；无编造/未解决诉求；SSE 与页面切回；真实 glm 项明确 `pending_upstream`。运行离线命令确认红，再实现最小评估器。
- [ ] 跑全量适用测试、迁移二次运行、离线评估、`git diff --check` 和关键浏览器操作；README 写可复制演示命令、日志敏感数据清理和真实验收待办。
- [ ] 请独立 reviewer 对 spec/plan/实现及五条 Review Focus 复核，修复问题后重跑受影响验证；逐阶段记 code review 结论与 finish。若原 Docker 业务卷未恢复，明确隔离库与生产库区别；若 glm 余额仍不足，不发真实模型请求。

## Handoff

用户已批准第 7 章书面 spec，尚未批准本计划或选择本章执行方式。计划通过后，按选定方式执行；鉴于任务共享数据库/Graph 接口且用户此前偏好节省额度，建议连续实施并在末尾独立复核。
