# PetShop_Helper 第 5 章 Workflow + ReAct Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有单轮客服编排迁入一个持久化的 LangGraph 流程，使知识路径强制检索、业务路径可多步用工具、用户点击后才建工单。

**Architecture:** 固定意图路由与证据闸包住有界 ReAct 环；规划节点不向 SSE 发送内容，独立最终答复节点负责逐 token 输出。MySQL 保留会话归属与业务审计，AsyncSqliteSaver 持久化图状态；两 API 共享图。

**Tech Stack:** Python 3.12、FastAPI、LangChain、LangGraph、langgraph-checkpoint-sqlite、aiosqlite、SQLAlchemy/MySQL、Milvus；前端沿用原生 HTML/CSS/JS。

**Spec:** [2026-09-26-ch05-workflow-react-design.md](../specs/2026-09-26-ch05-workflow-react-design.md)

## Global Constraints

- 上游、数据库、向量库与项目第 1–4 章的固定选型不变；glm-5.2 余额不足期间**不发付费模型请求**，真实验收标待补，不用别的模型冒充通过。
- 每项确定性代码按 Superpowers TDD 红→绿→提交；纯 Prompt/标注数据改用样例评估。聊天页面按用户要求 Vibe Coding，直接迭代，不套 brainstorm/TDD/code review。
- 具体 LangGraph、FastAPI、SQLAlchemy 接口先对照 [Context7 记录](../../context7-ch05.md)，安装版本首关用无模型冒烟确认；若固定选型在本机走不通，停下向用户说明，不自行替换。
- 当前会话归属规则必须保持；知识路径无可信 `sufficient`/引用即拒答；模型不得直接执行建单；单 worker，同会话并发返回 409。
- 每完成 brainstorm、计划评审、任务、review、finish 一个阶段，立即追记 `dev-notes/ch05.md` 的用户关键原话、产出、纠偏与返工；每个任务独立提交。

## Review Focus

1. 模型伪造 `create_ticket` 调用时不能写库：Task 5/7 的拒绝测试。
2. 分类或规划文本不能进入 SSE：Task 8 逐帧测试。
3. 旧 `query_faq` 的 `hits` 形状不能绕过证据闸：Task 4 缺字段测试。
4. 同一会话并发与另一用户猜测 conversation_id 不能交错或越权：Task 3/7/8 测试。
5. 同一 `request_id` 并发重试不能建两单，且建单不转人工：Task 6 MySQL 集成测试。

## File responsibilities

- `app/graph/state.py`：状态字段/归并；`routing.py`：七类映射和终止条件；`build.py`：节点与条件边；`nodes.py`：分类、检索、Agent、出口、日志；`runtime.py`：图生命周期、会话锁和两类调用。
- `app/core/intent.py`、`app/core/prompts.py`：单标签意图 Prompt 与第 5 章约束；`app/core/retrieval.py` 现有检索接口保持。
- `app/tools/registry.py`、`app/tools/infra.py`：按路径只绑定读工具并拒绝模型写工具；`app/db/repository.py`、`app/db/models.py`、`sql/ch05-ddl.sql`：不转人工的幂等建单。
- `app/api/agent.py`、`app/api/chat.py`、`app/api/actions.py`、`app/main.py`：共享图、SSE、动作端点与持久化生命周期；`app/static/index.html`：动作 UI。
- `scripts/eval_intent.py`、`scripts/eval_ch05.py`、`tests/graph/` 与 API/仓储测试：离线及后续真实验收。

## Task 1：依赖与无模型 fail-fast

**Files:** `pyproject.toml`、`uv.lock`、`scripts/smoke_ch05_graph.py`、`tests/graph/test_smoke.py`。
**Interfaces:** 产出本机可用的图、检查点和流形状；后续任务消费安装版本，不在此接业务。

- [ ] 先写本地冒烟断言：两节点条件边能结束；同一 `thread_id` 的第二次调用读取上次状态；`astream(..., stream_mode=["messages","updates"])` 产出 `(mode,payload)`，`messages` payload 是 `(message,metadata)`。
- [ ] 运行 `pytest -q tests/graph/test_smoke.py`，确认因缺依赖/实现而红。
- [ ] 锁定 `langgraph`、`langgraph-checkpoint-sqlite`、`aiosqlite` 的兼容版本并实现 `scripts/smoke_ch05_graph.py`；不调用聊天/嵌入/重排上游。
- [ ] 运行冒烟与测试；版本或 SQLite 连接不通时按固定选型排查，不跳到业务代码。
- [ ] 提交并立即追记任务结果。

## Task 2：状态与确定性路由

**Files:** `app/graph/state.py`、`app/graph/routing.py`、`app/core/intent.py`、`app/config.py`、`tests/graph/test_routing.py`。
**Interfaces:** `ConversationState` 包含 `messages` 的 add_messages 归并；`route_by_intent(intent: str) -> str`；`should_continue(state: ConversationState) -> str` 返回 `tools|final|fallback`；`classify_intent` 先提供可注入接口及失败关闭，Task 9 接真实 Prompt。

- [ ] 写七类→四出口、未知意图兜底、步数边界 5/6、每轮状态重置且历史保留的失败测试。
- [ ] 运行测试确认红；实现最小状态与路由，`max_agent_steps=6` 配置校验。
- [ ] 运行 `pytest -q tests/graph/test_routing.py` 及相关旧配置测试，确认绿。
- [ ] 提交并追记。

## Task 3：图生命周期、会话归属与串行化

**Files:** `app/graph/runtime.py`、`app/main.py`、`tests/graph/test_runtime.py`。
**Interfaces:** `GraphRuntime` 提供 `ainvoke_turn(user_id,message,conversation_id,model)` 与 `astream_turn(...)`；同会话同时请求抛出 `ConversationBusy`；lifespan 持有并关闭 checkpointer。

- [ ] 测试先红：新会话分配 MySQL ID；归属不符不读检查点；同一会话并发一方返回 busy；第二轮同 `thread_id` 保留消息；关闭后 SQLite 句柄释放。
- [ ] 实现单 worker 的 per-conversation 锁和所有权检查，`thread_id=str(conversation_id)`；用隔离 SQLite 文件和现有 `repository` 会话函数。图实例由可注入的 `build_graph` 工厂传入，Task 5 接业务图。
- [ ] 运行 `pytest -q tests/graph/test_runtime.py` 与会话相关测试确认绿。
- [ ] 提交并追记单 worker 运行约束。

## Task 4：知识路强制检索与弱证据闸

**Files:** `app/graph/nodes.py`、`tests/graph/test_knowledge_route.py`。
**Interfaces:** `forced_rag(state) -> {sufficient,evidence,citations,reason}` 的状态更新；`confidence_gate(state) -> agent|fallback`。使用当前 `bm25_query`/`get_settings()` 接口。

- [ ] 测试先红：强证据只引用已编号原文；无命中、缺 `sufficient` 的旧 `hits`、异常或自评失败均不进入 Agent，低置信记录仅一次。
- [ ] 实现第 4 章查询/自评结果规范化和失败关闭，保持现有 citation 编号；不重写检索算法。
- [ ] 运行 `pytest -q tests/graph/test_knowledge_route.py` 及第 4 章检索测试确认绿。
- [ ] 提交并追记。

## Task 5：有界 ReAct 与工具权限

**Files:** `app/graph/nodes.py`、`app/graph/build.py`、`app/tools/registry.py`、`app/tools/infra.py`、`app/core/prompts.py`、`tests/graph/test_react.py`。
**Interfaces:** 业务模型工具=`query_order/query_product/query_logistics`；知识模型工具=`query_order`；`create_ticket` 不能被模型执行；最终答复在单独 `final_answer` 节点。

- [ ] 测试先红：假模型先查订单再查物流、观察第一次工具结果才第二次决策；第 6 步停止；伪造 `create_ticket` 被拒绝且 tickets 不变；知识路拒绝 `query_product/query_faq`；最终节点禁用工具。
- [ ] 改 `AGENT_SYSTEM` 的单轮/直接建单规则，构建 `agent_llm↔agent_tools` 环和四出口图；不改 `query_logistics(order_id)` 契约。
- [ ] 运行 `pytest -q tests/graph/test_react.py`、现有工具权限/编排测试，按新合同迁移旧断言后确认绿。
- [ ] 提交并追记旧接口迁移及模型调用次数取舍。

## Task 6：幂等建单且不隐式转人工

**Files:** `sql/ch05-ddl.sql`、`app/db/models.py`、`app/db/repository.py`、`app/api/actions.py`、`tests/test_ch05_actions.py`。
**Interfaces:** `create_ticket_only(conversation_id,description,ticket_type,request_id) -> str`；`POST /api/actions/create-ticket` 校验 `user_id` 与会话归属，返回 `ticket_no`。

- [ ] 测试先红：重复相同 ID 返回同单号；同 ID 不同参数冲突；并发只一张票；另一用户 404；会话状态不变；空白内容 422。
- [ ] 增加 `tickets.request_id` 唯一列迁移，并实现事务性仅建单；唯一约束冲突后回查原请求，不能在失败事务继续查询。将旧 `repository.create_ticket` 的转人工副作用保留给旧兼容场景，不供新端点调用。
- [ ] 在隔离 MySQL 执行 DDL/集成测试，运行 `pytest -q tests/test_ch05_actions.py` 与第 2 章工单测试；若迁移与既有数据冲突先排查。
- [ ] 提交并追记。

## Task 7：非流式 API 共享图

**Files:** `app/api/agent.py`、`app/schemas/agent.py`、`app/main.py`、`tests/test_ch05_agent_api.py`。
**Interfaces:** `/api/agent` 仍返回 `conversation_id/answer/tool_calls/tool_results`，新增 `suggested_actions`；会话 busy=409，越权=404，模型故障按既有错误边界。

- [ ] 写失败测试：已有字段兼容、同会话续接、投诉动作不建票、伪造写工具不建票、busy 和越权响应。
- [ ] 接 `GraphRuntime.ainvoke_turn`，保留依赖注入假模型和错误边界；不再从 API 调用旧单轮编排。
- [ ] 运行 `pytest -q tests/test_ch05_agent_api.py tests/test_agent_api.py` 确认绿。
- [ ] 提交并追记。

## Task 8：SSE 只透出最终答复

**Files:** `app/api/chat.py`、`app/graph/runtime.py`、`tests/test_ch05_chat_stream.py`。
**Interfaces:** 既有 `delta/tool/citations/done` 帧保持；新增 `actions`；`delta` 仅来自 `final_answer`，固定出口独立发文案。

- [ ] 写失败测试：分类/自评/规划/工具调用片段不在任何 `delta`；最终答复分块逐帧；投诉 actions 两项；故障有 error 帧且不假报 done；越权/并发路径无泄漏。
- [ ] 用 Context7 已核对的 `(mode,payload)` 流形状按 `langgraph_node` 过滤，接现有 SSE 生成器并处理取消。
- [ ] 运行 `pytest -q tests/test_ch05_chat_stream.py tests/test_chat_api.py tests/test_agent_stream.py` 与本地 socket 流检查确认绿。
- [ ] 提交并追记。

## Task 9：意图 Prompt 与标注评估

**Files:** `app/core/intent.py`、`app/core/prompts.py`、`tests/data/intent_ch05.jsonl`、`scripts/eval_intent.py`、`tests/test_eval_intent_ch05.py`。
**Interfaces:** 七类固定标签，结构化输出；分类失败转安全兜底，不进业务工具。

- [ ] 先整理每类标注样例（含投诉与退款退货易混项），用数据校验脚本检查七类覆盖、重复和标签拼写；纯 Prompt/数据不用机械 TDD。
- [ ] 实现分类适配与离线假模型路径测试；在额度不足时只跑格式/路径校验，将真实分类正确率标 `pending_upstream`。
- [ ] 运行样例校验和路径测试；真实模型恢复时补跑 `scripts/eval_intent.py` 并记录分桶混淆情况。
- [ ] 提交并追记 Prompt 评估范围和未完成部分。

## Task 10：聊天页面动作与端到端收尾

**Files:** `app/static/index.html`、`scripts/eval_ch05.py`、`Makefile`、`README.md`、`dev-notes/ch05.md`。
**Interfaces:** 接 `actions` SSE，独立转人工/建工单按钮与确认表单；离线五路径报告区分 `passed_offline`、`pending_upstream`。

- [ ] 聊天页以 Vibe Coding 直接改并浏览器迭代：投诉动作可见、取消不写、确认后返回单号且禁用、继续对话可用；不为页面另套 brainstorm/TDD/review。
- [ ] 用假模型/隔离库做知识强弱、物流多步、闲聊、投诉、跨轮五路径离线集成；重跑 62 个测试文件的分组全量回归并核对第 4 章无回归。
- [ ] 真实 glm-5.2 额度不足时不调用；明确列出无法完成的真实 SSE/JSON 与标注意图验收，保留续跑命令和日志路径。完成代码复核后处理重要发现。
- [ ] 更新 README 与 `dev-notes/ch05.md`，提交；按 Superpowers finishing-a-development-branch 处理分支，不把未验收内容说成完成。

## Self-review

- Spec 的图、状态、四出口、证据闸、步数、工具权限、动作隔离、两 API、SSE、持久化、并发和验证均有对应任务。
- Task 1/3 提供图与持久化，Task 2/4/5 消费状态，Task 7/8 消费 runtime，Task 6/8 提供动作 API/UI；接口名称一致。
- 真实模型验收受已知余额问题阻断，不被离线任务误标完成；后续章节只能在保留该缺口的情况下继续设计/实施。
