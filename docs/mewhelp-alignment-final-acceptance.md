# MewHelp 对齐：最终阶段记录

后续上传说明（2026-09-28）：Git身份现已配置为hannnz1，并按用户选择创建私有PetShop_Helper远程。下文为功能验收时点记录；上传快照不改变IV真实验收尚未完成的结论。

2026-09-28，批准计划的I/II/III功能补齐已实施；IV真实验收与主目录集成尚未完成。这里记录实际证据，不将离线通过等同全量交付。

## 当前可用成果

消息反馈归属/失败重试；证据置信度校准与回退检索；型号校验；作业鉴权；用量与趋势；审核/发布/向量重试；清洗与增强人工审核；分类诊断/服务元数据/持久化；主题统计分页；九项验收后台；统一导航和逐章样例。

最新代码在 `C:/Users/Administrator/Documents/Codex/2026-09-23/kai/PetShop_Helper/.worktrees/ch08-tool-system`。该工作树基线abe481e、分支feat/ch09-observability；主目录9c36b00尚未包含后续工作。Git作者未配置，产品改动尚未提交/合并，不能把上层旧应用当本次版本。

## 验证证据

| 检查组 | 结果 | 范围与限制 |
|---|---|---|
| 对齐集成 | 249 passed，153.74秒 | alignment/ch10/flywheel/observability及聊天、作业、知识相关回归；1个第三方弃用警告 |
| 配置修复 | 39 passed，12.78秒 | PyMilvus不隐式读取祖先.env、配置与预算；先复现2失败/1通过 |
| 旧章节回归 | 195 passed，194.03秒 | graph/tools/db、退款、中断、三层上下文、工具确认、记忆、提取；修复前32失败/163通过 |
| 最终分类版本链修复 | 44 passed，7.54秒 | acceptance API＋全部ch10；两项新测试先RED后GREEN |
| 静态检查 | 通过 | git diff --check（CRLF提醒）、3个后台JS语法、uv lock --check --offline |
| 当前代码启动 | 通过 | start-alignment启动app.main于8772；health/逐章页200，使用离线上游和隔离MySQL |
| 浏览器 | 合成验证通过 | 反馈、四种审核、向量失败重试、主题22题分页、缺失报告明确展示 |

上述测试组存在重叠，不相加为唯一测试数。未重复运行收费全量评估；本轮真实模型调用为0。集成原始日志保存在本地忽略目录 `.superpowers/sdd/2026-09-28-mewhelp-alignment-plan/`。

独立复核已修复增强待审核、并发跳过计数、黄金闸预标版本、分类与审计原子性，以及训练→ONNX版本链问题。清除凭据时延迟响应体的epoch保护已实现，正常清除/跨页浏览器验证通过；人为延迟响应体场景的浏览器补验仍待补，未声称覆盖。

截图：[反馈重试](evidence/alignment/feedback-retry.png)、[审核向量重试](evidence/alignment/review-vector-retry.png)、[合成主题](evidence/alignment/topics-synthetic.png)、[第2页](evidence/alignment/topics-page2-synthetic.png)、[缺失验收](evidence/alignment/acceptance-missing.png)、[最新样例入口](evidence/alignment/manual-samples-entry.png)。

## 未完成与下一步

| 待办 | 现场阻塞 | 恢复后下一步 |
|---|---|---|
| IV-1四策略全量/校准 | 正式MySQL3306、Milvus19530未运行；只有3307隔离库 | 先核对原卷/备份、恢复权威知识；运行smoke_milvus_bm25与小样例；校验缓存指纹后续跑，不加--fresh |
| 真实知识回流 | 缺真实向量服务 | 审核发布→向量重试→真实查询命中；不能只看SQL done |
| 真实观测对账 | Langfuse容器在运行，但本轮无真实模型trace | 小样例后对比usage/trace，不为凑趋势跑付费全量 |
| IV-2分类真实验收 | 无data/ch10训练产物，ML依赖未装，C盘约1GB | 将算力/模型数据安置于充足磁盘；准备审核语料，确认第10章真实语料外发授权；按IV-2执行黄金闸/划分/训练/阈值/ONNX/服务/落库 |
| 主目录集成 | Git user.name/email未配置 | 使用用户实际作者身份提交，核对两checkout后集成；当前保留全部未提交文件 |

旧生成/检索缓存6个文件已非覆盖复制到当前work/ch04-eval-cache；未取得权威库不能承诺当前可复用数量。原报告partial保留。完整300题四策略为1200项，无调用错误与质量达标是两种不同判定。恢复收费验收遇首个确定余额错误应停止该批；不能将普通限流重试当余额恢复。

第10章CLI实际smoke_service写入pending_compute（无模型包）并非0退出；没有生成假passed报告。真实报告必须同时绑定数据、预标配置、训练checkpoint、ONNX版本及术语，过期状态显示stale。

## 使用与演示

先按[部署说明](DEPLOY.md)恢复服务与配置，再运行：

```powershell
cd C:\Users\Administrator\Documents\Codex\2026-09-23\kai\PetShop_Helper\.worktrees\ch08-tool-system
.\scripts\start-alignment.ps1 -EnvFile ..\..\.env
# 第二终端：真实模型/服务就绪后，会产生上游调用
.\.venv\Scripts\python.exe -X utf8 -m scripts.demo_chat --base-url http://127.0.0.1:8000
```

浏览器打开 `/admin`、`/manual-test-samples` 查看各功能入口、可复制问题及模拟/真实边界。分类恢复命令见[分类验收](mewhelp-alignment-classifier-acceptance.md)。开发过程按阶段记录于[dev-notes](../dev-notes/mewhelp-alignment.md)，逐项状态见[21项执行状态](mewhelp-alignment-status.md)。
