# 第 10 章主题分类：离线验收记录

状态：**离线实现与独立复核已通过；真实模型验收未完成**。本文件只记录当前仓库的结果，不引用课程原版历史 F1 作为本仓库成绩。

## 本地复演

在项目 worktree 根目录、已配置基础 Python 虚拟环境后运行：

```powershell
$env:CHAT_MODEL='offline-test'
$env:CHAT_BASE_URL='http://127.0.0.1:9/v1'
$env:CHAT_API_KEY='offline-test'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.venv/Scripts/python.exe -X utf8 -m pytest -q -p pytest_asyncio.plugin tests/ch10 tests/observability/test_config.py tests/observability/test_tracing.py
.venv/Scripts/python.exe -X utf8 -m scripts.ch10.build_corpus --synthetic-only --out data/ch10/synthetic-corpus
.venv/Scripts/python.exe -X utf8 -m scripts.ch10.build_dataset --source data/ch10/synthetic-corpus/corpus_labeled.jsonl --out data/ch10/synthetic-dataset --include-supplement
.venv/Scripts/python.exe -X utf8 -m scripts.ch10.train --dataset data/ch10/synthetic-dataset --out data/ch10/model
.venv/Scripts/python.exe -X utf8 -m scripts.ch10.export_onnx --model data/ch10/model --test data/ch10/synthetic-dataset/test.jsonl --out data/ch10/onnx
```

2026-09-28 额度恢复后，按上面的测试命令重跑：**46 passed、1 个第三方弃用警告**（Starlette/AnyIO）。`git diff --check` 退出码为 0。之前自动审批因额度上限拒绝执行，以及修复过程中一次旧阈值断言失败，均已由本次复跑覆盖。

审查修复后重新导出合成语料 30 条黄金样例；加 34 条尺码补充后，切分为 train=62、val=1、test=1。每个原始类别不足 100 条，数据集与训练 CLI 均返回 `pending_data`，ONNX 导出返回 `pending_compute`。重新生成的清单包含分割哈希和审核 ID。

## 人工审核与真实验收

真实低置信问题只能在本地脱敏后进入审核队列；外部聊天模型调用需另行授权。`build_corpus` 产生带 `review_id` 的 `corpus_labeled.jsonl` 与抽审清单 `audit.md`。预标失败的真实条目仍保留脱敏文本、来源 ID 和失败状态，供人工给出标签，不自动归“其他”。审核者另写 JSONL 决议，例如：

```json
{"review_id":"pool-17","approved":true,"labels":["退换货"]}
{"review_id":"simulated-<导出文件中的实际ID>","approved":false}
```

然后运行：

```powershell
.venv/Scripts/python.exe -X utf8 -m scripts.ch10.review_corpus --source data/ch10/corpus_labeled.jsonl --decisions data/ch10/review_decisions.jsonl --out data/ch10/corpus_reviewed.jsonl
.venv/Scripts/python.exe -X utf8 -m scripts.ch10.build_dataset --source data/ch10/corpus_reviewed.jsonl --out data/ch10/dataset --include-supplement
```

未通过人工审核的行不进入数据集；数据集输出保留来源与审核状态，并将每个切分的内容哈希写入清单。训练前必须验证这些哈希，导出前必须验证测试集哈希。

真实验收待办：

- `pending_upstream`：真实问题的外部预标、修错、造数、改写没有本章的单独外发授权；当前聊天上游额度此前不足。若改用本地模型，仍须通过黄金样例 80% 集合全对闸。
- `pending_data`：真实问题缺审核决议；合成语料各类远低于 100 条，不能据此报告训练质量。
- `pending_compute`：可选训练依赖与模型权重未安装/下载；尚无本仓库训练模型、ONNX 完整测试集一致性报告或真实 `127.0.0.1:8110/classify` 结果。
- 第 9 章 MySQL 回归：本机 `127.0.0.1:3307` 拒绝连接，DB 相关测试未完成；第 10 章纯离线测试不依赖该库。
- 独立复核结论：修复后未发现剩余 Important/Critical 代码问题；离线命令已复跑。真实模型和第 9 章数据库回归仍按各自前置条件待验收。

有真实模型且 `export_report.json` 为 `passed` 后，安装独立 `ch10-serving` 依赖组，再用 `CH10_MODEL_DIR=data/ch10/onnx` 启动 `uvicorn scripts.ch10.serve:app --host 127.0.0.1 --port 8110`，以 `curl -X POST http://127.0.0.1:8110/classify -H 'Content-Type: application/json' -d '{"texts":["猫窝买大了怎么办"]}'` 验证 17 类分数。测试集还须达到严档 F1 ≥ 0.9、中档 F1 ≥ 0.8，并通过验证阈值回放。
