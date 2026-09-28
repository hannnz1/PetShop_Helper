# MewHelp 基线、反馈与检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在最新版本接通真实反馈，补齐可校准检索闸与评估型号校验。

**Architecture:** 复用 Graph 持久化标记和现有反馈仓储。抽出小型 FAQ 管线供工具与强制 RAG 选择不同闸门，纯计算与上游调用分开。

**Tech Stack:** FastAPI、LangGraph、LangChain、SQLAlchemy/MySQL、Milvus、pytest、HTML/JS。

**Spec:** [补充设计 §3](../specs/2026-09-28-mewhelp-feature-alignment-design.md)。执行约定见[总计划](2026-09-28-mewhelp-alignment-plan.md)。

## Global Constraints

- 沿用最新 worktree 和固定技术栈；查询 Context7 并核对锁定版本后实现接口。
- 新 SSE 字段可选且兼容旧客户端；中断未完成不得伪造可反馈 ID；聊天页面用 Vibe Coding。
- 置信度权重 0.5/0.2/0.2/0.1，有效证据线 0.3、数量封顶3；只输入 rerank 分。
- 校准 0.05–0.95 步长0.01，最大 Youden J，平分保留较低阈值；没有有效校准继续旧闸。
- 型号机械闸仅评估路径，最多一次修复；不得暗改实时 SSE 或退款政策规则。

## Review Focus

1. 旧气泡绑定新会话：I-2 验证捕获气泡自身 ID。
2. 数据库保存失败却生成 done：I-2 断言无可反馈完成帧。
3. NaN/Infinity 污染置信度：I-3 断言拒绝非法分。
4. 语义 category 回退撤销硬约束：I-4 断言只撤 category。
5. 机械校验修复失败却沿用旧裁判 PASS：I-5 保留失败并重新评估最终答案。

## 文件边界

新建 `app/core/evidence_confidence.py`（纯计算/配置报告验证）、`app/core/faq_pipeline.py`（查询到证据）、`app/core/model_guard.py`（型号集合）、`scripts/ch09/calibrate_confidence.py`（批次校准）。修改现有 chat/nodes/tools，保持外部工具 schema 不变。测试放 `tests/alignment/`，现有测试复用而不整体搬迁。

### I-1：运行基线和交付状态（A01/A02 的准备部分）

**Files:** 修改 `README.md`、补充 spec 状态、`docs/mewhelp-feature-alignment-checklist.md`；新建 `docs/mewhelp-alignment-status.md`。
**Interfaces:** 状态表每行 `{id, implementation, verification, evidence, blocker}`；本任务不新增业务 API。

- [ ] 核对 `git status --short`、HEAD、已有 worktree 和本地配置文件存在性；记录使用路径，不输出配置值。
- [ ] 按已有 dev-notes 和报告填 21 项状态及 1–10 章真实/离线状态；校正有证据的旧标题/题数，无法确定批准历史则标注。
- [ ] 在 README 给出当前唯一使用入口与基础启动命令；只探活必要服务，不自动重启/重建卷。
- [ ] 验证本地链接、文档路径与实际 HEAD；留痕并提交本任务文件。A01/A02 最终关闭仍依赖 IV-3。

### I-2：真实回答 ID 到反馈按钮（F01）

**Files:** 修改 `app/api/chat.py`、`app/static/index.html`；必要时修改 `app/graph/state.py`；新增 `tests/alignment/test_feedback_stream.py`，复用 `tests/flywheel/test_feedback_api.py`。
**Interfaces:** `graph_event_stream(graph_stream, user_id)` 从 log_turn 更新中的 `trace.audit_message_id` 读取 ID；done 帧增加 `assistant_message_id: int`。现有 `record_unresolved_feedback(user_id, conversation_id, assistant_message_id)->int` 保持。

**Test anchors:** `test_done_uses_committed_audit_id`: `assert done['assistant_message_id'] == 42`；`test_interrupted_turn_has_no_feedback_id`: `assert not any('assistant_message_id' in frame for frame in frames)`。

- [ ] RED：构造 audit_message_id=42，断言 done.assistant_message_id==42；中断、保存异常断言没有该完成帧；历史返回的各消息 ID 保持原值。
- [ ] 运行 `Test-Alignment tests/alignment/test_feedback_stream.py`，确认新增字段缺失导致失败。
- [ ] 修改 SSE 转换，不增加 latest 查询；前端以气泡的 ID/user/conversation 发送 POST，成功后才锁定，失败可重试；无 ID 不开放发送。👍 明示本地评价。
- [ ] 运行 `Test-Alignment tests/alignment/test_feedback_stream.py tests/flywheel/test_feedback_api.py tests/test_ch06_resume_api.py`；在隔离库浏览器验证新/历史/切换会话气泡、重复点击和网络失败，保存 API+SQL 证据。
- [ ] 立即写留痕、更新状态并提交相关文件；浏览器与后端未全部验证时 F01 不关闭。

### I-3：置信度纯计算与校准报告（R01 计算部分）

**Files:** 新建 `app/core/evidence_confidence.py`、`scripts/ch09/calibrate_confidence.py`、`tests/alignment/test_confidence.py`、`tests/alignment/test_calibration.py`；修改 `app/config.py`、`.env.example`。
**Interfaces:** `compute_evidence_confidence(hits: list[dict])->dict` 返回 score/signals；`calibrate(rows: list[dict])->dict` 输入 bucket/score；`load_calibration(path: Path, expected: dict)->dict` 返回 ready/pending 和有效阈值。配置 `EVIDENCE_CALIBRATION_PATH` 指向本地报告，不设置参考阈值默认值。

**Test anchors:** `test_four_signal_score`: `assert result['score'] == .5733`；`test_missing_negative_bucket_is_pending`: `assert report['status'] == 'pending_calibration'`。

- [ ] RED：两条无关键字的 .8/.6 分，score==.5733；空证据==0；nan/inf 拒绝；校准平分保留更低线；缺 D 桶或可答桶返回不可校准；报告版本不符返回 pending。
- [ ] 运行 `Test-Alignment tests/alignment/test_confidence.py tests/alignment/test_calibration.py`，确认缺模块/行为断言失败。
- [ ] 实现固定公式、阈值扫描和版本校验；CLI `python -m scripts.ch09.calibrate_confidence --scores-file <固定分数JSONL> --out <报告>` 可离线重演，`--live` 才调用检索；保存 dataset/strategy/signal/retrieval-config 指纹，输出 recommended/in_use 信息。
- [ ] GREEN 同一测试命令；重复对固定分数表运行 CLI，断言扫描结果相同且未调用聊天模型。真实校准留 IV-1。
- [ ] 留痕、更新计算部分状态并提交；未校准不标 R01 全部完成。

### I-4：检索管线接闸与 category 回退（R01/R02）

**Files:** 新建 `app/core/faq_pipeline.py`、`tests/alignment/test_faq_gates.py`；修改 `app/tools/business.py`、`app/graph/nodes.py`，复用 `tests/test_query_faq_rag.py`、`tests/graph/test_knowledge_route.py`。
**Interfaces:** `async run_faq_pipeline(keyword: str, category: str|None=None, *, gate: str='top1')->dict`，gate 仅 `top1|calibrated`；返回现有 sufficient/source/reason/evidence/citations，另含 trace 信号。工具调用 top1，forced_rag 调 calibrated；无有效校准 fallback 到旧 top1 并记录 pending。

**Test anchors:** `test_wrong_category_retries_once`: `assert categories == ['猜错的分类', None]`；`test_confidence_rejection_precedes_selfcheck`: `assert selfcheck_calls == 0`。

- [ ] RED：category 猜错仅重试一次且第二次 category=None；有效校准的弱证据在 selfcheck 调用前拒绝；没有校准保持旧行为；低置信只入池一次；退款 retrieve_policy 行为不变。
- [ ] 运行 `Test-Alignment tests/alignment/test_faq_gates.py tests/graph/test_knowledge_route.py`，确认针对性 RED。
- [ ] 抽出当前查询理解→混合检索→回退→机械闸→selfcheck→证据组装；保持 Lite 分支兼容但不把其分数当 rerank 校准；forced_rag 复用纯服务而不先调用已自检的工具。保留唯一入池责任在 Graph。
- [ ] GREEN：`Test-Alignment tests/alignment/test_faq_gates.py tests/test_query_faq_rag.py tests/graph/test_knowledge_route.py tests/graph/test_ch06_policy.py`；核对单轮调用计数和 trace。
- [ ] 留痕并提交；记录 R01 真实校准未完成状态。

### I-5：评估答案型号校验（R03）

**Files:** 新建 `app/core/model_guard.py`、`tests/alignment/test_model_guard.py`；修改 `scripts/eval_ch04.py`、`tests/test_eval_ch04.py`。
**Interfaces:** `unsupported_models(answer: str, evidence: str)->list[str]`、`repair_hint(bad: list[str])->str`；生成记录增加 original_answer、unsupported_models、repair_status、guard_passed，缓存指纹包含新算法/Prompt 版本。

**Test anchors:** `test_unsupported_model_is_mechanical`: `assert unsupported_models('MH-CAD1', 'MH-CAM1') == ['MH-CAD1']`；`test_failed_repair_does_not_retry`: `assert repair_calls == 1`。

- [ ] RED：MH-CAM1 证据不能支持 MH-CAD1；大小写不一致不通过；正常文本不误拦；第一次修复失败后调用次数不再增加；裁判评估最终答案并保留机械失败状态。
- [ ] 运行 `Test-Alignment tests/alignment/test_model_guard.py tests/test_eval_ch04.py`，确认新增场景失败。
- [ ] 实现确定性比对和最多一次模型修复；失败记报告和个案，不能覆盖为成功；只失效受改变影响的生成缓存，检索缓存规则保持。
- [ ] GREEN 同一命令，使用假生成器验证请求次数；真实样例随 IV-1，不在本任务额外跑全量收费批次。
- [ ] 留痕、提交；批次 I 运行以上相关测试一次，保存 `docs/mewhelp-alignment-core-acceptance.md` 并注明未完成真实项。
