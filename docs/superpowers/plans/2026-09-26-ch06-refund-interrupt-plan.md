# Chapter 6 Refund Interrupt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让客服能消解短指代、暂停等待本人订单选择、基于政策证据给出退款申请入口，并在用户确认后持久记录待审核申请。

**Architecture:** 在第 5 章单 worker 图上增加指代、取单和政策节点；合法 `select_order` 中断走同一 SQLite thread 的 `Command(resume=...)`。MySQL 保存演示订单和退款申请；模型工具只提出动作，唯一写入路径是用户确认后的 FastAPI 端点。

**Tech Stack:** Python 3.12、FastAPI、LangChain、LangGraph、SQLAlchemy 2.0/MySQL、Milvus、原生 HTML/CSS/JS。

**Spec:** [第 6 章已批准设计](../specs/2026-09-26-ch06-refund-interrupt-design.md)。库接口先按 [Context7 记录](../../context7-ch06.md)，本机行为以无模型冒烟为准。

## Global Constraints

- 不更换 FastAPI、LangChain、LangGraph、MySQL、Milvus 和 OpenAI 兼容模型配置；不调用余额不足的真实 glm 上游。
- 样例订单是演示数据；真实退款、支付、自动审批和正式身份认证不在本章。`user_id` 延续现有演示身份边界。
- 模型无权写退款申请；申请状态只为“待人工审核”。`request_id` 相同内容幂等、不同内容 409。
- 原始用户消息写 MySQL；`resolved_query` 仅用于理解/检索。证据不足不提供 `refund_form`。
- 只支持单 worker；普通新消息不能吞掉 pending 选单中断；第 5 章异常审计分歧仍拒绝续聊。
- 所有确定性代码先红后绿；纯 Prompt/样例改用标注集评估。聊天页面按用户指定的 Vibe Coding 直接改。
- 每完成一项立即追记 `dev-notes/ch06.md` 的用户原话、产出/路径、纠偏、翻车返工；不要收尾集中补。

## File Map

- `scripts/smoke_ch06_interrupt.py`：纯 LangGraph 中断/续跑/流形状冒烟。
- `sql/ch06-ddl.sql`、`scripts/migrate_ch06.py`、`scripts/seed_ch06_demo.py`、`app/db/models.py`、`app/db/repository.py`：演示订单与退款申请持久层；种子显式执行。
- `app/core/coref.py`、`app/core/query_understanding.py`、`app/core/prompts.py`：指代消解与政策多查询扩写。
- `app/graph/state.py`、`app/graph/routing.py`、`app/graph/build.py`、`app/graph/nodes.py`：退款图状态、出口、节点与证据闸。
- `app/tools/refunds.py`、`app/tools/registry.py`、`app/tools/infra.py`：建议入口工具；当前项目没有 `builtin/` 子包，沿用现有工具目录；执行分发层仍禁止模型写库。
- `app/graph/runtime.py`、`app/api/chat.py`、`app/api/agent.py`、`app/api/actions.py`：中断帧、续跑预检与确认写入。
- `app/static/index.html`：沿用现有内嵌聊天脚本与样式，做选单/退款表单。
- `scripts/eval_ch06.py`、`tests/data/ch06_samples.jsonl`：离线四路径与标注样例报告。

## Review Focus

1. 用户提交别人的 `order_id` 或篡改选单值 → 不泄露他人订单、不发申请入口；Task 4/8 复测。
2. 没有样例订单的用户进入退款意图 → 明确兜底，不能无限中断；Task 4 复测。
3. 旧会话没有 checkpoint 或 pending 类型不是 `select_order` → 返回明确错误，不执行普通消息/错误续跑；Task 7 复测。
4. 检索有命中但政策不足以判断 → 不生成退款表单；Task 5/6 复测。
5. 相同 `request_id` 携带不同订单/原因重试 → 409，原申请保持不变；Task 2/8 复测。

---

### Task 1: 钉死本机中断合同

**Files:** Create `scripts/smoke_ch06_interrupt.py`; Test `tests/graph/test_interrupt_smoke.py`; update `docs/context7-ch06.md` 与 `dev-notes/ch06.md`。

**Interfaces:** Produces documented local shapes for `interrupt` update, `Command(resume=str)`, node rerun, `aget_state().next` and cross-runtime checkpoint restore; no business API.

- [x] 写纯图测试：`ainvoke` 得到 `__interrupt__`；同 `thread_id` 的 `Command(resume="1001")` 得到选单值；节点前只读计数在续跑时增长；`astream(messages+updates)` 可识别中断。
- [x] 用本机 `.venv` 跑 `pytest -q tests/graph/test_interrupt_smoke.py`，先确认缺脚本/行为断言失败。
- [x] 实现最小纯图冒烟，脚本不建业务库、不调用模型。
- [x] 运行测试与脚本并把实际载荷写入 Context7 记录；不符合文档时停下定位而非猜测。
- [x] 提交代码、测试、阶段笔记。

### Task 2: 演示订单与申请持久层

**Files:** Create `sql/ch06-ddl.sql`, `scripts/migrate_ch06.py`, `scripts/seed_ch06_demo.py`, `tests/test_ch06_repository.py`; modify `app/db/models.py`, `app/db/repository.py`, `dev-notes/ch06.md`。

**Interfaces:** `list_sample_orders(user_id: str) -> list[dict]`; `get_owned_sample_order(user_id: str, order_id: str) -> dict | None`; `create_refund_request(conversation_id: int, user_id: str, order_id: str, reason: str, request_id: str) -> str`; raises `RefundRequestConflict` on same key/different payload.

- [x] 写隔离 MySQL 失败测试：本人/他人订单隔离、无订单空列表、提交前零写入、相同请求幂等、参数变化 409、申请状态“待人工审核”。
- [x] 跑 `pytest -q tests/test_ch06_repository.py` 确认红。
- [x] 用非破坏性 DDL/迁移建 `sample_orders`、`refund_requests`，事务性仓储写入；种子脚本显式导入固定演示用户/订单，不在应用启动时自动播种。
- [x] 跑定向测试和迁移二次运行；确认不触碰现有工单及业务记录。
- [x] 提交并追记迁移结果。

### Task 3: 指代消解与短历史

**Files:** Create `app/core/coref.py`, `tests/test_ch06_coref.py`; modify `app/core/prompts.py`, `app/graph/state.py`, `app/graph/nodes.py`, `app/core/intent.py` if classifier input needs unchanged API, `dev-notes/ch06.md`。

**Interfaces:** `resolve(query: str, history: str, model: BaseChatModel) -> str`; graph stores `resolved_query`, keeps original `query`; classifier receives resolved text.

- [x] 写失败测试：完整句保意、"那它能退吗"从最近历史补订单、空答/模型异常回原句、当前问题不作为自己的历史、MySQL 审计仍用原文。
- [x] 跑 `pytest -q tests/test_ch06_coref.py` 确认红。
- [x] 增加 LangChain Prompt 和最多 6 轮完整消息构建；在 `coref` 节点调用并传给分类/检索，不修改原始用户消息。
- [x] 跑核心图与 API 回归，检查 token 预算和异常回退。
- [x] 提交并追记标注样例验证（Prompt 效果不以单测代替）。

### Task 4: 本人订单节点与选择中断

**Files:** Create `tests/graph/test_ch06_fetch_order.py`; modify `app/graph/state.py`, `app/graph/routing.py`, `app/graph/build.py`, `app/graph/nodes.py`, `dev-notes/ch06.md`。

**Interfaces:** `fetch_order(state: ConversationState) -> dict`; emits `interrupt({"type":"select_order","orders": ...})`; success returns `order_id`, `order_data`; `退款退货/售后` route to `refund`.

- [x] 写编译图失败测试：`订单1001` 可抽号；缺号中断；他人号与他人 resume 值再次中断；零订单兜底；续跑后只有本人订单快照。
- [x] 跑 `pytest -q tests/graph/test_ch06_fetch_order.py` 确认红。
- [x] 实现数字 lookaround 抽号、持久归属查询、只读前置 `interrupt`；不直接调用节点去测试 interrupt。
- [x] 跑测试与第 5 章路由测试；旧 `售后` 出口断言按批准设计更新。
- [x] 提交并追记。

### Task 5: 退款政策检索与证据闸

**Files:** Create `tests/graph/test_ch06_policy.py`; modify `app/core/query_understanding.py`, `app/core/prompts.py`, `app/graph/build.py`, `app/graph/nodes.py`, `dev-notes/ch06.md`。

**Interfaces:** `expand_queries(query: str, model: BaseChatModel) -> list[str]`，最多 3 条非空，失败 `[query]`；`retrieve_policy(state, runtime) -> dict` 统一为 `sufficient/evidence/citations/reason`。

- [x] 写失败测试：订单状态进检索种子、最多三查、同 chunk 取最高分、扩写失败回原查询、弱证据/错误格式不进入 Agent 且无退款动作。
- [x] 跑 `pytest -q tests/graph/test_ch06_policy.py` 确认红。
- [x] 沿用第 4/5 章 hybrid-rerank/充分性契约组装退款政策证据；模型 Prompt 只用于查询扩写，不更改库中原文。
- [x] 跑定向及现有 RAG 证据闸测试。
- [x] 提交并追记证据不足结果。

### Task 6: 模型建议退款表单，不写申请

**Files:** Create `app/tools/refunds.py`, `tests/graph/test_ch06_refund_action.py`; modify `app/tools/registry.py`, `app/tools/infra.py`, `app/graph/nodes.py`, `app/core/prompts.py`, `dev-notes/ch06.md`。

**Interfaces:** `submit_refund(order_id: str, user_id: InjectedToolArg, reason: str | None = None) -> dict` 仅返回待用户确认；`agent_tools` 拦截同名调用并输出 `suggested_actions=[{"type":"refund_form","draft":...}]`。

- [x] 写失败测试：模型提出退款但数据库不变；本人订单+强证据才有表单；他人订单/错号/弱证据无表单；重复建议不重复动作；最终答复不称已退款。
- [x] 跑 `pytest -q tests/graph/test_ch06_refund_action.py` 确认红。
- [x] 将建议工具只绑定在 refund 路径，图拦截前重校验归属和本轮订单；可信分发层拒绝 `submit_refund` 实际写入，旧 `create_ticket` 禁止规则保持。
- [x] 跑定向测试及旧工具安全回归。
- [x] 提交并追记。

### Task 7: 图续跑及 SSE 中断帧

**Files:** Create `tests/graph/test_ch06_resume.py`, `tests/test_ch06_resume_api.py`; modify `app/graph/runtime.py`, `app/api/chat.py`, `app/api/actions.py`, `dev-notes/ch06.md`。

**Interfaces:** `GraphRuntime.prepare_resume_turn(user_id: str, conversation_id: int, order_id: str, *, model: BaseChatModel) -> AsyncIterator[tuple[str, Any]]`; `POST /api/actions/resume` 返回 SSE；`interrupt` 帧含 `kind`, `conversation_id`, `orders`，无 `done`。

- [x] 写失败测试：合法缺单→中断→同会话恢复；新消息撞 pending 返回 409；旧会话无检查点/其他 pending 不可冒充选单；越权 404、忙碌 409，均在 HTTP 200 前；正常已完成不可再 resume。
- [x] 跑定向测试确认红。
- [x] 结合 Task 1 本机载荷修改审计预检：仅识别 `select_order` 合法 pending 且 MySQL 标记一致；续跑以相同 thread 的 `Command(resume=order_id)` 运行，共用 SSE 帧转换。
- [x] 跑定向、跨接口续聊和本地 socket 冒烟；异常不发成功 `done`。
- [x] 提交并追记。

### Task 8: 确认后创建退款申请

**Files:** Create `tests/test_ch06_refund_api.py`; modify `app/api/actions.py`, `dev-notes/ch06.md`。

**Interfaces:** `POST /api/actions/create-refund` 请求 `user_id/conversation_id/order_id/reason/request_id`，返回 `refund_no/status`；固定五种原因。

- [x] 写失败测试：合法确认一条待审核申请；他人会话/订单 404；空/非法原因 422；重复同 payload 同申请号；相同 ID 不同订单/原因 409；不改变订单和工单状态。
- [x] 跑 `pytest -q tests/test_ch06_refund_api.py` 确认红。
- [x] 在 FastAPI 请求模型校验原因并调用 Task 2 仓储；数据库再校验用户与订单归属，不能只靠页面/模型。
- [x] 跑定向和 Ch05 建工单 API 回归。
- [x] 提交并追记。

### Task 9: 聊天页面交互（Vibe Coding 例外）

**Files:** Modify `app/static/index.html`, `dev-notes/ch06.md`。

**Interfaces:** `interrupt(select_order)` 渲染本人订单卡并调用 resume；`actions(refund_form)` 渲染固定原因下拉和确认/取消；只有确认才发 `create-refund`。

- [x] 直接修改现有页面并用离线假 SSE/内存动作服务浏览器演示：缺单选单续跑、取消无 POST、确认显示申请号、重复按钮禁用、错误订单不泄露。
- [x] 页面 UI 按用户约定不套 brainstorm、TDD、code review；后端保护仍由 Task 7/8 测试。
- [x] 提交并逐项追记实际浏览器观察与返工。

### Task 10: 四路径验收与独立复核

**Files:** Create `scripts/eval_ch06.py`, `tests/data/ch06_samples.jsonl`; modify `README.md`, `dev-notes/ch06.md`。

**Interfaces:** 报告明确区分 `passed_offline`、`pending_upstream`，覆盖指代、缺单续跑、错号重选、确认写入四路径；真实模型不自动触发。

- [x] 标注样例先验证 Prompt/数据：记录指代保真与扩写不编造；假模型/隔离库运行四条端到端路径并写报告。
- [x] 跑相关全量回归（Windows 原生扩展若单进程崩溃，按文件独立进程跑并如实标注）；跑 `git diff --check`。
- [x] 请求一次独立代码复核，修复重要发现后再复测；不调用付费模型。
- [x] README 写无模型演示、迁移/样例种子、真实服务 curl、测试结果及第 4/5/6 章待上游边界。
- [ ] 追记 code review 结论与 finish 四项记录，提交最终文档。
