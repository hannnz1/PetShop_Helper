# MewHelp 真实验收与交付 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用本项目真实数据/模型验证功能，交付可启动、可演示、状态可信的最终版本。

**Architecture:** 消费前三批实现与报告，先低成本依赖和小样本验证，再补失败项和真实训练。使用独立报告目录与基线标识，完成后做集成和独立复核。

**Tech Stack:** 现有 FastAPI/Graph/MySQL/Milvus/Langfuse、固定模型上游、ch10 ml/ch10-serving依赖组、PowerShell。

**Spec:** [补充设计 §6–8](../specs/2026-09-28-mewhelp-feature-alignment-design.md)；执行约定和映射见[总计划](2026-09-28-mewhelp-alignment-plan.md)。

## Global Constraints

- 300题×4策略=1200条，缓存指纹匹配才复用；不使用 --fresh 无差别重跑；调用错误为0才算报告完整，业务质量另判。
- 真实文本外发按既有授权范围；未获ch10独立授权不能自动增加 `--allow-external-real-text`。
- 黄金预标≥80%，每类原始样本≥100；严格类F1≥.9、中档≥.8；验证阈值.30–.70步长.05，平分保留先到线；全测试集Torch/ONNX标签mismatch=0。
- 数据库验收用隔离库与合成操作数据；不得把重置正式库作为演示步骤。不可用项明确pending，不反复盲重试收费请求。
- Git 作者未配置不自行伪造；最终集成先检查两checkout改动和依赖，再选择能保留工作的集成方式。

## Review Focus

1. 余额不足返回429被当作限速不断重试：IV-1 首个确定余额错误停止该付费批次。
2. Prompt变更后重复用旧答案缓存：IV-1 核对指纹与最终答案校验状态。
3. 为达到F1把难题补进测试集或训练集泄漏：IV-2 固定holdout并保存hash。
4. 训练和ONNX用不同阈值/术语：IV-2 全量版本链及重演核验。
5. worktree通过但交付入口启动旧代码：IV-3 两路径HEAD和一次最终入口烟测。

## 文件边界

本批主要更新 `docs/mewhelp-alignment-final-acceptance.md`、`docs/mewhelp-alignment-status.md`、`README.md`、`docs/DEPLOY.md`、`docs/mewhelp-feature-alignment-checklist.md`；新增演示样例页。已有脚本缺能力时回到对应I/II/III任务修复，不在验收任务夹带大改。

### IV-1：真实第1–9章与检索/飞轮验收（R04/W02/O02及回归）

**Files:** 新建/更新最终验收报告；复用 `scripts/eval_ch04.py`、`scripts/ch09/calibrate_confidence.py`、`scripts/ch09/eval_trend.py`、既有 eval_ch05/06/07/08 及演示脚本。
**Interfaces:** 每个验收记录 `{chapter,git_sha,dataset_hash,model,command,status,evidence,blocker}`；原始报告保存在忽略的 data/ 下，版本化摘要进入docs。

- [ ] 在真实配置终端检查必要服务、迁移版本和上游可用性；最小授权样例成功才扩批，余额不足停止付费部分并继续其它本地验收。记录实际状态，不沿用旧服务结论。
- [ ] 运行 `python -m scripts.validate_eval_ch04`，确认300题标注合法；核对已有474成功缓存当前哪些仍有效，记录预计补跑数量。按CLI现有入口 `python -m scripts.eval_ch04 --generation-concurrency 3` 续跑，不加--fresh；CLI部分报告必须非零退出，最终1200条完整且指标分母可对账。
- [ ] 运行 `python -m scripts.ch09.calibrate_confidence --live --out data/ch09/reports/confidence_calibration.json`，核对标注桶映射和推荐/生效版本；用浏览器样例确认知识强/弱路径、错category回退、引用/拒答入池。
- [ ] 在隔离库浏览器完整执行F01/W02：两种同义问句入池→标准化→人工核准→发布→向量化→再次检索；故意中断一次向量操作并恢复，SQL检查无重复。人工审核只用合成知识。
- [ ] 根据清单逐章复验：ch01真实SSE/两轮/结构化；ch02工具及会话持久化；ch03双写/重跑；ch05意图/有界循环；ch06选单/重启恢复/确认才写；ch07真实多轮摘要和usage预算；ch08 MCP/预览/取消/超时审计。先读取对应脚本 `--help`，不猜测支持的参数，选择其真实模式并把精确命令写入报告。
- [ ] 本机Langfuse对一组明确request/run ID核对实际usage；缺usage显示不可用。用 `python -m scripts.ch09.eval_trend --live` 形成真实趋势样本时遵守预算；该脚本禁用ch04缓存，不能为凑“两点”无条件重复全量，缺第二点记录待验收。
- [ ] 汇总所有实测结果、错误分母和未完成原因，更新清单；留痕并提交文档。没有真实证据不关闭相应条目。

### IV-2：第10章真实语料、训练、导出与分类（C07）

**Files:** 输出 `data/ch10/` 版本产物；更新 `docs/ch10-topic-classifier-acceptance.md` 和最终报告。既有脚本不因验收改变模型或阈值标准。
**Interfaces:** corpus→reviewed→dataset→model→onnx→8110→topic_classifications；每步校验上一步hash/status，失败停止依赖步骤。

- [ ] 确认真正可用的数据、算力/磁盘、ml依赖组和固定模型权重；核对外发授权。已有授权不足只请求缺失范围，不重复索要密钥；其余离线任务照常完成。
- [ ] 在许可满足后运行 `python -m scripts.ch10.build_corpus --out data/ch10/corpus`，外部真实文本仅在获准后加已有授权开关；保存修错样例逐条质量结果和黄金闸报告，低于80%不做批量预标；导出审核决定并用 `python -m scripts.ch10.review_corpus --source data/ch10/corpus/corpus_labeled.jsonl --decisions data/ch10/review_decisions.jsonl --out data/ch10/corpus_reviewed.jsonl` 导入。
- [ ] 运行 `python -m scripts.ch10.build_dataset --source data/ch10/corpus_reviewed.jsonl --out data/ch10/dataset --include-supplement --augment`，相应外发开关仅在授权时加；核对每类原始≥100、val/test非空、固定split hash和增强来源。pending_data时补审核/训练数据，不动已固定考卷抬分。
- [ ] 运行 `python -m scripts.ch10.train --dataset data/ch10/dataset --out data/ch10/model`；保存最优权重、阈值和验证分数。然后 `python -m scripts.ch10.export_onnx --model data/ch10/model --test data/ch10/dataset/test.jsonl --out data/ch10/onnx`，全测试集最终标签含兜底一致才继续。
- [ ] 运行 `python -m scripts.ch10.evaluate --model data/ch10/onnx --test data/ch10/dataset/test.jsonl --out data/ch10/evaluation.json` 和 `python -m scripts.ch10.scan_threshold_replay --model data/ch10/onnx --val data/ch10/dataset/val.jsonl --out data/ch10/threshold_replay.json`；核对红线和同一分数表阈值一致。未达标保留失败报告，回到训练数据任务。
- [ ] 以 `CH10_MODEL_DIR=data/ch10/onnx` 启动 `python -m uvicorn scripts.ch10.serve:app --host 127.0.0.1 --port 8110`；健康/真实classify返回同版本17类分数；在隔离池运行 `python -m scripts.ch10.classify_pool --limit 500 --min-batch 10` 两遍，第一遍落库、第二遍empty，主题页面与SQL对账。
- [ ] 九项验收页与CLI同源一致，保存真实截图/模型清单/报告；留痕、提交验收摘要，不把大权重或真实语料提交Git。

### IV-3：统一导航、演示、部署和最终复核（A01/A02/D01/D02）

**Files:** 修改 `README.md`、`app/api/admin.py`、`app/static/admin.html`、`app/main.py`、清单与状态表；新建 `docs/DEPLOY.md`、`app/static/manual-test-samples.html`、`tests/alignment/test_admin_navigation.py`。
**Interfaces:** 首页入口覆盖chat/kb/rag-eval/review/observability/topics/acceptance/manual-test-samples；部署说明逐服务给出当前平台确实验证的命令，调度默认未启用。

- [ ] 后端RED检查导航目标都有实际HTML路由，未知URL不伪返回成功；运行 `Test-Alignment tests/alignment/test_admin_navigation.py`。
- [ ] 实现统一导航与逐章可复制问句/预期行为/模拟或真实标注；部署文档写明配置变量名、迁移顺序、端口、启动/停止、备份与失败续跑；复用现有Windows调度说明，默认不创建付费定时任务。
- [ ] GREEN导航测试；从目标入口按部署说明复演一次；合并/快进前检查主目录和worktree改动、运行进程及Git关系，保留未提交内容。若存在实际冲突先解决并相关复验，不重命名worktree来冒充已集成。
- [ ] 运行分组的最终离线/数据库/本机集成测试，各进程保存结果；真实报告引用IV-1/2，不因页面改动重跑所有收费评估。独立复核实现与spec/21项清单；按receiving-code-review处理发现，修复后只重跑相关范围。
- [ ] 所有ID均有实现/测试/真实证据才更新完成；任何pending列明具体阻塞与下一条命令。最终报告交付功能演示命令、测试结果、最新使用入口和dev-notes路径；完成记录即刻写入，不事后补造阶段日志。
