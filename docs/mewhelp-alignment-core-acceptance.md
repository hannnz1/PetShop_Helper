# 批次 I 验收记录

2026-09-28，基线 abe481e 上未提交工作；全计划尚未完成。

- SSE绑定已提交消息ID，失败/中断无可反馈完成帧。隔离MySQL反馈及恢复8项通过；之后补了显式中断测试。
- 四信号置信度、91个阈值扫描、最低同分阈值、配置/数据/代码指纹，离线回放与线上改写一致性通过。未执行真实校准。
- 工具top1、Graph calibrated；缺报告维持旧闸，分类仅撤销一次。18项FAQ/Graph/退款/Lite回归通过。
- 型号最多一次修复，裁判消费最终答案；机械失败覆盖错误语义PASS。生成缓存独立失效；旧检索缓存只按当前知识/配置精确迁移，原文件保留。
- 批次离线集成60项通过；随后缓存迁移等相关27项通过。独立review问题已修复，复核无其余high/medium。
- 浏览器：反馈首轮模拟502后可重试，成功锁按钮；新会话切回历史重复提交，SQL仍1条且指向原会话。[SQL](evidence/alignment/sql.json) / [截图](evidence/alignment/feedback-retry.png)。固定回复不作为真实模型质量验收。

离线演示：`python -m scripts.ch09.calibrate_confidence --scores-file <分数JSONL> --out <报告JSON>`。绑定标注集须额外给 `--dataset <标注JSONL> --retrieval-config-hash <匹配指纹>`，ID/桶完整匹配。未绑定报告不能启用到运行时。

尚待真实校准/评估、最终集成和依赖复演；本阶段无新付费模型调用。
