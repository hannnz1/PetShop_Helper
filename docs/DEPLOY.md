# 当前版本部署与恢复

上传阶段更新（2026-09-28）：Git作者已配置，私有远程为https://github.com/hannnz1/PetShop_Helper。下文Git身份缺失描述保留原验收时点背景；远程开发快照与本机主目录整合是两件事。新机器克隆后应进入克隆目录，无需照搬本机绝对worktree路径。

最新代码目录：`C:/Users/Administrator/Documents/Codex/2026-09-23/kai/PetShop_Helper/.worktrees/ch08-tool-system`。主 checkout 仍为 9c36b00，不能从上层旧入口验证本轮功能。当前分支基线 abe481e，补齐改动未提交（Git 作者身份未配置）。保留两个 checkout，未执行 merge/reset。

## Windows 启动

先在最新目录执行一次 `uv sync --locked` 安装基础依赖。已有虚拟环境不需要每次同步。真实环境须恢复 MySQL/知识向量库，按下节迁移，并检查配置。当前现场只验证了离线模型＋隔离 MySQL 的启动和页面，真实环境依赖缺失见最终验收记录。

```powershell
cd C:\Users\Administrator\Documents\Codex\2026-09-23\kai\PetShop_Helper\.worktrees\ch08-tool-system
.\scripts\start-alignment.ps1 -EnvFile ..\..\.env
```

该入口明确启动当前目录，复用指定 .env，默认启动本机 MCP、前台监听 `127.0.0.1:8000`。不复制或打印密钥、不自动建库或安装大模型。按 Ctrl+C 停止应用；`./scripts/mcp.ps1 -Action Stop` 只停止该脚本记录的 MCP 进程。`-Port 8772 -SkipMcp` 可用于单独检查页面。

| 配置 | 用途 |
|---|---|
| CHAT_BASE_URL / CHAT_MODEL / CHAT_API_KEY | 上游 OpenAI 协议聊天 |
| DATABASE_URL / TEST_DATABASE_URL | 正式 schema / 独立 *_test schema，不能相同 |
| SILICONFLOW_API_KEY / EMBED_BASE_URL / EMBED_MODEL | BGE-M3 嵌入 |
| MILVUS_URI / RERANK_BASE_URL / RERANK_MODEL | 四策略检索与重排 |
| KNOWLEDGE_REVIEW_TOKEN | 审核、主题及分类作业凭据 |
| OBSERVABILITY_ADMIN_TOKEN | 用量、校准、分类验收凭据 |
| CLASSIFIER_BASE_URL | 默认 http://127.0.0.1:8110，只允许本机 HTTP |
| LANGFUSE_ENABLED / LANGFUSE_BASE_URL / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY | 自托管观测 |

按原已批准上下文预算设置 MODEL_CONTEXT_WINDOW 等参数；若旧配置仍设置 TOKEN_BUDGET=2000，启动预算自检会拒绝不足的窗口，应按实际模型上下文能力配置，不能只为启动伪填更大窗口。当前离线验收使用 TOKEN_BUDGET=32768。

## 服务与迁移

| 服务 | 默认端口 | 本轮现场状态 |
|---|---|---|
| 应用 | 8000 | 当前代码可启动；真实依赖待恢复 |
| MySQL | 3306 | 正式服务不存在；3307仅隔离验收库 |
| Milvus Standalone | 19530 | 未运行 |
| MCP物流/售后 | 8101 / 8102 | 由 mcp.ps1 管理 |
| 分类服务 | 8110 | 缺本项目模型包，不可用 |
| Langfuse | 3000 | 容器已运行；本轮未发送真实模型 trace |

先核对现有数据库卷/备份来源再恢复正式 MySQL，不把空库或测试库当正式数据。`docker compose up -d mysql` 是仓库已有基础启动命令，但首次空库初始化与已有卷恢复的结果不同。不要执行 `docker compose down -v` 或 reset 知识库来处理连接错误。当前 Langfuse MinIO 已占9091，仓库 Milvus health端口也映射9091；同时部署前须明确调整其中一个宿主端口及对应健康探测，不能直接假定两套 compose 均可启动。

本轮新增迁移在配置目标库后运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.ch10.migrate_topics
```

只创建缺失的分类结果表和批次表，重复执行不清数据，已在隔离 MySQL 验证。前置为第2–9章 schema；旧库缺第9章飞轮表时先执行 `python -m scripts.ch09.migrate_flywheel`。第5/6/7章已有迁移分别为 scripts.migrate_ch05 / migrate_ch06 / migrate_ch07。第8/9章其余增量 SQL 应先与 information_schema 对照，不能无条件重放所有 ALTER。`scripts.migrate_ch04` 会重建向量，不属于普通 SQL 迁移，不自动调用。

迁移 CLI 从当前 .env/进程环境取配置；`start-alignment -EnvFile` 参数仅用于启动进程。运行迁移前可将已核对配置保存为当前目录 .env（若已存在就编辑，不覆盖），或在终端设置配置变量；勿将密钥写进命令历史。

## 演示入口

打开 `/admin` 或 `/manual-test-samples`。聊天 `/`、知识 `/kb`、RAG `/rag-eval`、审核 `/review`、用量 `/observability`、主题 `/topics`、验收 `/acceptance` 均有真实 HTML 路由。审核/观测凭据仅留当前页面内存，刷新/离开/清除后不保留。

```powershell
# 配置与真实服务就绪后执行；会调用模型
.\.venv\Scripts\python.exe -X utf8 -m scripts.demo_chat --base-url http://127.0.0.1:8000
curl.exe -N http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" --data-raw '{"user_id":"demo","message":"我养了一只三岁的猫"}'
curl.exe http://127.0.0.1:8000/api/extract -H "Content-Type: application/json" --data-raw '{"text":"订单 MH20260701123 的猫爬架散架了，我要退款"}'
```

PowerShell版本对引号传递有差异时，用已有 demo_chat.ps1 或将 JSON 保存 UTF-8 文件后使用 `curl.exe --data-binary @文件名`。第二轮带上第一轮返回的 conversation_id。

## 分类流水线与恢复

真实语料外发需第10章独立授权；本轮未借用300题授权。按已批准顺序执行 build_corpus（默认 data/ch10/corpus）、review_corpus、build_dataset、train、export_onnx、evaluate、scan_threshold_replay、serve、smoke_service、classify_pool。具体完整参数见实施计划 IV-2。

增强首次 `build_dataset --augment` 会输出待审核变体（reviewed=false），不能训练；为每个 `aug-...` review_id 写 `{"review_id":"...","approved":true}` 或 false 的 JSONL，使用同一源文件和缓存再加 `--augmentation-decisions data/ch10/augmentation-decisions.jsonl`。只改变 train，val/test 不重分。待补数据、未审核、上游失败均不作 ready。类别原始样本至少100，严格/中档红线保持原标准。

训练依赖组 `uv sync --locked --group ml` 与轻服务 `--group ch10-serving` 分开；未安装时应显示 pending_compute。本轮C盘约1GB可用，未下载权重或安装训练依赖。先落实数据、算力与足够磁盘，再安装到确定的环境。

```powershell
$env:CH10_MODEL_DIR='data/ch10/onnx'
.\.venv\Scripts\python.exe -m uvicorn scripts.ch10.serve:app --host 127.0.0.1 --port 8110
# 第二个终端
.\.venv\Scripts\python.exe -m scripts.ch10.smoke_service
.\.venv\Scripts\python.exe -m scripts.ch10.classify_pool --limit 500 --min-batch 10
```

ONNX报告必须匹配当前包指纹；旧passed报告需重新导出验证。分类失败不写空标签；成功分类与批次成功状态同事务，异常留running/failed审计；重跑未分类项不重复，升级模型需显式 --reclassify。批次报告保留在 data/ch10/classification-runs/，最新计数兼容文件为 pool_categories.json。

第4章已将旧缓存非覆盖保留到当前 work/ch04-eval-cache。恢复权威 MySQL 后先核对缓存指纹，改写/Prompt/证据变化不得直接复用旧生成答案；不加 --fresh。先最小样例确认额度，余额错误停止收费验收。校准需真实标注集与检索配置一致，未就绪时保留旧门控。

审核发布成功但向量失败：从审核详情重试“全部待向量知识”，不会重复建知识块；重试可能调用嵌入接口。恢复后还须实际检索命中才关闭 W02。

## 调度与备份

默认不创建任何 Windows 计划任务，不启用每日收费全量评估。后续可手动为 classify_pool 本机批次与飞轮待审核队列导出配置任务计划程序，工作目录须为当前 worktree、禁止并行重复实例、日志写独立目录；外发/训练开关不从网页参数任意传入。涉及收费的 eval_trend 禁用缓存，不能为凑趋势数据自动重复全量。

备份 MySQL数据卷/逻辑备份、Graph checkpoint及待审核状态、忽略目录 work/ch04-eval-cache 与 data/ch10。真实语料、权重、密钥不进Git。清理磁盘前先确认备份可恢复；本轮没有清理这些资料。
