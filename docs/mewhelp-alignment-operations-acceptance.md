# 批次 II 验收记录

2026-09-28，代码已实施，真实向量与观测对账仍待验。

- II-1：统一新作业鉴权 RED11失败/3通过→GREEN25通过。旧kb总览隐藏受保护新作业。校准需观测凭据和heavy确认，未实际启动。
- II-2：缺usage保留null，平均数分母为available_calls，区块独立失败、说明绑定报告。18项汇总/隔离DB趋势通过；后补默认评估集缺失不隐藏用量测试，待集成复跑。
- II-3：详情遮蔽来源，发布状态派生自knowledge行，全局pending向量重试明确确认。34项审核/发布/知识库回归通过。
- II-4：页面路由/数据鉴权18项通过；浏览器approve/defer/merge/reject与SQL对账；单块向量模拟失败→重试成功，审核决定保留。观测空用量为无数据，缺校准为pending；刷新审核页凭据为空，清除观测凭据会清空报告。
- 独立review指出响应体解析期间清除凭据仍可能回填、缺评估集导致整页503；已加解析后epoch检查、数据集读取移至趋势区块。延迟响应体的浏览器补验仍待完成。

证据：[SQL](evidence/alignment/sql.json)、[向量重试](evidence/alignment/review-vector-retry.png)、[观测](evidence/alignment/observability-empty.png)。

隔离演示：`python -m uvicorn scripts.acceptance.alignment_ui:app --host 127.0.0.1 --port 8771`。配置本地3307专用 `mewhelp_alignment_ui_test` 和测试review/observability凭据。脚本只接受该库，不清空已有数据；聊天/向量/作业均为替身，首次反馈/向量调用故意失败。不能据此宣称真实Milvus、Langfuse和模型语义达标。
