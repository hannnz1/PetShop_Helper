# 第 9 章评估趋势：单次运行与定期调度

`scripts.ch09.eval_trend` 每次读取第 4 章标注集，计算其 SHA-256，把一份不可覆盖的原始报告保存到 Git 忽略的 `data/ch09/eval-runs/`，并向 MySQL `eval_runs` 写四个策略的摘要。在已经应用 Task 3 `sql/ch09-observability.sql` 的数据库上，单独执行 `sql/ch09-eval-runs.sql`；不要重跑 Task 3 的一次性 `ALTER TABLE`。真实运行还需要第 4 章的 Milvus、SiliconFlow 和有可用额度的聊天模型；缺依赖时只记录 `pending_upstream`，不能据此判断模型质量。真实运行禁用第 4 章缓存，每次重新获取指标。

PowerShell 中切换到项目工作树后，手动运行一次：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.ch09.eval_trend --live
```

`make eval-ch09` 调用同一命令；Windows 没有 GNU Make 时直接执行上述 Python 命令。只验证标注报告解析且不调用上游时，先准备本地报告，再执行 `python -m scripts.ch09.eval_trend --report-file <本地报告路径>`。这种导入被标记为 `pending_upstream`，其数字指标入账时置为不可用，不会进入真实可比趋势。

Windows 任务计划程序可每日触发一次 `powershell.exe`，参数为 `-NoProfile -Command "Set-Location '<工作树绝对路径>'; .\.venv\Scripts\python.exe -X utf8 -m scripts.ch09.eval_trend --live"`。建议仅在依赖服务和模型额度均恢复后启用。任务计划程序运行账户必须能访问本机 `.env` 和 MySQL；不要把密钥写进任务参数或日志。

趋势 API：`GET /api/observability/eval-trend?dataset_hash=<64位SHA256>&strategy=vector`，请求头为 `Authorization: Bearer <OBSERVABILITY_ADMIN_TOKEN>`。相同数据集但指标分母变化的记录标为 `comparable=false`；不同数据集 hash 需分别查询，不应画成同一条曲线。报告的 `passed` 只表示当次指标数据完整，不代表自动达到业务质量阈值。
