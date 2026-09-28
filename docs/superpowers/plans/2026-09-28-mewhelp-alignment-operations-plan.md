# MewHelp 运营后台 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付可操作且受保护的观测、人工审核和知识回流后台。

**Architecture:** 页面消费现有仓储/审核服务，新增轻量 overview 和详情接口。所有新作业共享权限分发，读图数字来自同一报告。

**Tech Stack:** FastAPI、SQLAlchemy/MySQL、Langfuse、JobRunner、HTML/JS。

**Spec:** [补充设计 §4](../specs/2026-09-28-mewhelp-feature-alignment-design.md)；依赖 [I-3 校准报告](2026-09-28-mewhelp-alignment-core-plan.md)。测试约定见[总计划](2026-09-28-mewhelp-alignment-plan.md)。

## Global Constraints

- 新审核/主题作业用 KNOWLEDGE_REVIEW_TOKEN，观测/验收作业用 OBSERVABILITY_ADMIN_TOKEN；未配置503、缺/错凭据401。
- token 仅页面内存，不进入 URL/持久存储/静态资源/日志。报告读取不调用收费模型。
- 新作业只接受白名单参数，脚本未落地前不注册；所有启动/停止/状态/日志入口执行同一权限规则。
- MySQL 为真实 usage 权威账；缺 usage 不填零；没有价目表只报告 token。
- approved_pending_vector 不等于可检索；不得自动发布模型示例答案。

## Review Focus

1. 调 `/api/jobs/{name}` 绕过新页面：II-1 全入口鉴权矩阵。
2. 分母为零或全部 usage 缺失：II-2 均值/占比 null，不伪零。
3. 一块依赖坏导致全部后台不可用：II-2 区块独立错误测试。
4. 正式答案审核后向量失败：II-3 保留决定、状态 pending、重试幂等。
5. 刷新页面把 token 或用户原文留在 storage：II-4 浏览器检查。

## 文件边界

新增 `app/core/admin_access.py`（新作业权限）、`app/observability/dashboard.py`（报表汇总）、`app/observability/read_notes.py`（确定性说明）、`app/static/operations.js`（共用交互）；沿用现有 flywheel/review，避免把页面逻辑塞进 repository。

### II-1：新作业统一权限（O01/W01/C06 基础）

**Files:** 新建 `app/core/admin_access.py`、`tests/alignment/test_job_access.py`；修改 `app/core/jobs.py`、`app/api/jobs.py`、`Makefile`。
**Interfaces:** `JobSpec.permission: str|None` 取 review/observability/None；`require_job_access(name: str, credentials: str|None, settings)->None`；列表对无权的新作业不暴露状态/日志，其它出口拒绝。

**Test anchor:** `test_sensitive_job_rejects_legacy_route_bypass` 参数化各操作：`assert response.status_code == 401; assert runner_calls == 0`。

- [ ] RED：参数化 GET详情、GET日志所用状态、POST启动、POST停止和列表；未配503、错误401、正确才到 runner；未知作业404；新作业不能经旧路由绕过。
- [ ] 运行 `Test-Alignment tests/alignment/test_job_access.py`，确认新增权限断言失败。
- [ ] 实现统一分发；配置/脚本与 job 同任务落地才注册。已完成校准脚本可注册 calibration；后续任务自己注册对应新作业，禁止接受自由命令/路径。
- [ ] GREEN：`Test-Alignment tests/alignment/test_job_access.py tests/test_jobs_api.py`；验证日志脱敏与原有重作业确认语义仍成立。
- [ ] 留痕并提交；后续新增 job 必须追加同一参数化权限测试列表。

### II-2：观测汇总、用量与读图说明（O01/O02/O03 后端）

**Files:** 新建 `app/observability/dashboard.py`、`app/observability/read_notes.py`、`tests/alignment/test_observability_dashboard.py`；修改 `app/api/observability.py`、`scripts/ch09/usage_report.py`。
**Interfaces:** `summarize_usage(rows: list[dict])->dict` 返回 measured_tokens/calls/available_calls/unavailable_calls/avg_tokens/token_share；`build_read_note(kind: str, report: dict)->dict` 返回 report_id/text；`async build_overview(start, end, dataset_hash, strategy)->dict` 返回 cost/trend/calibration 独立区块。增加受保护 GET `/api/observability/overview`，原 API 保持。

**Test anchor:** `test_missing_usage_is_not_zero`: `assert summary['avg_tokens'] == 150; assert summary['unavailable_calls'] == 1`。

- [ ] RED：100+200 token 两次可用调用、一次 unavailable，断言均值150、缺失1且不是100；零总量 share=None；hash或分母不同不可比较；坏校准文件不影响账；注释 report_id 不符不展示。
- [ ] 运行 `Test-Alignment tests/alignment/test_observability_dashboard.py`。
- [ ] 实现仓储汇总和模板说明；API/CLI 用同一函数，规定均值分母为 available_calls；保留模型/意图/日期维度及 request/run 关联。模型小注默认关闭，本计划以确定性说明完成等效展示。
- [ ] GREEN：`Test-Alignment tests/alignment/test_observability_dashboard.py tests/observability/test_usage_report.py tests/observability/test_eval_trend.py`；固定 fixture CLI/API 输出相同数字；真实 Langfuse 对账交 IV-1。
- [ ] 留痕、提交，说明 O02 真实对账仍待验收。

### II-3：审核详情与发布可恢复状态（W01/W02 后端）

**Files:** 修改 `app/flywheel/review.py`、`app/api/flywheel.py`；新建 `app/kb/vectorization_service.py`、`tests/alignment/test_review_status.py`；修改 `app/api/kb.py` 共用既有向量化流程，不从服务层调用 API 函数。
**Interfaces:** `async get_review_detail(canonical_id: int)->dict|None` 返回遮蔽来源/审计/knowledge_chunk_id/vector_status；`async vectorize_pending_knowledge()->int` 复用当前 client、锁和 dualwrite；新增受保护 GET `/api/review/questions/{id}`、POST `/{id}/vectorize`。重试作业是全局 pending 批次时页面明确范围并要求确认。

**Test anchor:** `test_vector_failure_preserves_approved_decision`: `assert row.status == 'approved_pending_vector'; assert knowledge_count == 1`；后续重试成功仍 `assert knowledge_count == 1`。

- [ ] RED：未批准不能触发关联发布；待向量不显示 ready；嵌入失败决定保留；成功后状态读取真实 knowledge 行；重复批准/发布不重复；详情不泄露电话号码。
- [ ] 在隔离 MySQL 运行 `Test-Alignment tests/alignment/test_review_status.py tests/flywheel/test_publish.py`，确认是行为 RED 而非连接失败。
- [ ] 实现只读派生发布状态及受保护向量重试，不把 vector done 伪写回审核决定；操作审计沿用原请求幂等。
- [ ] GREEN 同一命令并跑 `tests/flywheel/test_review.py`；用可用隔离 Milvus 验“失败后重试→可检索”，不可用则明确 pending，不关 W02。
- [ ] 留痕、提交，记录实际用了 fake embedding 还是真实服务。

### II-4：审核与观测页面（O01/W01）

**Files:** 新建 `app/static/review.html`、`app/static/observability.html`、`app/static/operations.js`、`tests/alignment/test_operations_routes.py`；修改 `app/main.py`、`app/api/admin.py`、`app/static/admin.html`。
**Interfaces:** `/review` 消费 II-3 与现有 decision/publish；`/observability` 消费 II-2；凭据仅各页内存。角色分别匹配 II-1，不建立新用户系统。

**Test anchor:** `test_operations_html_does_not_unlock_private_api`: 页面 `assert response.status_code == 200`，无token数据请求 `assert api_response.status_code == 401`。

- [ ] 后端 RED：新页面路由返回404，新增后返回HTML；页面数据依然无凭据401/未配置503。运行 `Test-Alignment tests/alignment/test_operations_routes.py`。
- [ ] 实现列表/详情/审核表单/发布状态、成本/趋势/校准三区块、受保护重跑按钮和明确失败提示；等待操作时防双击，request_id 在同一次重试中复用。
- [ ] GREEN 上述路由测试与 II-1；真实浏览器操作 approve/merge/defer/reject、发布失败再重试，确认页面数字来自同一报告。
- [ ] 浏览器验证刷新不残留 token、失败不误显示成功、无旧报告冒充最新、后台首页可进入两页；保存截图和接口响应摘要。
- [ ] 留痕、提交；写 `docs/mewhelp-alignment-operations-acceptance.md`，清楚列真实向量/语义/对账待办。
