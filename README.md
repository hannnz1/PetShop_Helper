# PetShop_Helper：电商智能客服（第 1–3 章开发中）

FastAPI 服务通过 OpenAI 兼容的 **Chat Completions** 接口连接模型。`/api/chat` 返回带工具轨迹的 SSE 流，使用 MySQL 保存多轮会话；`/api/agent` 返回完整 JSON 及工具轨迹；`/api/extract` 将售后描述提取为固定字段。第 2 章包括五个 LangChain 工具：三个模拟数据查询、FAQ 查询和工单创建。第 3 章正在把 FAQ 内部查询切换为 BGE-M3 + Milvus Lite 的 dense 语义检索，并建设 `/kb` 知识库工作台。订单、商品、物流查询结果均为演示数据。

## 配置与启动

需要 Python 3.12、[uv](https://docs.astral.sh/uv/)、Docker Desktop 和支持 tool calling 的 OpenAI 兼容模型服务。复制 `.env.example` 为 `.env`，填写真实的 `CHAT_BASE_URL`、`CHAT_MODEL`、`CHAT_API_KEY`，并确认 `DATABASE_URL` 指向本地 MySQL。第 2 章已用 `glm-5.2` 做真实 tool-call 冒烟与端到端评估；其他模型也必须先单独验证 tool calling。默认 `STRUCTURED_OUTPUT_METHOD=json_mode`，供应商支持时可设为 `json_schema`。密钥由 Git 忽略，勿提交。

Bash（第一个终端）：

```bash
cp .env.example .env
# 编辑 .env，填写真实配置
docker compose up -d mysql
make seed
./scripts/dev.sh
```

Windows PowerShell（第一个终端）：

```powershell
Copy-Item .env.example .env
# 编辑 .env，填写真实配置
docker compose up -d mysql
.\scripts\seed.ps1
.\scripts\dev.ps1
```

服务监听 `http://127.0.0.1:8000`；浏览器打开此地址即可使用聊天页。`GET /health` 应返回 `{"status":"ok"}`。`make dev` 是启动入口。

## 验收

在服务运行时，于第二个终端执行。第 1 章提取与多轮流式演示仍可运行；第 2 章评估通过真实接口调用当前模型和本地 MySQL。

| 项目 | Bash | Windows PowerShell |
|---|---|---|
| 五条标注提取样例 | `uv run --locked python scripts/eval_extract.py` 或 `make eval` | `.\.venv\Scripts\python.exe -X utf8 scripts\eval_extract.py` |
| 两轮流式对话 | `./scripts/demo_chat.sh` | `.\scripts\demo_chat.ps1` |
| 客服与提取行为样例 | `uv run --locked python scripts/eval_prompts.py` | `.\.venv\Scripts\python.exe -X utf8 scripts\eval_prompts.py` |
| 第 2 章九条选工具评估 | `make eval-agent` | `.\.venv\Scripts\python.exe -X utf8 scripts\eval_agent.py` |
| 第 2 章三项轨迹演示 | `./scripts/demo_agent.sh` | `.\scripts\demo_agent.ps1` |

提取样例来自 `tests/data/extract_samples.json`。`eval_extract.py` 比较 `order_id` 与 `request_type`；`expected_solution` 需人工核对。`demo_chat.py` 验证 SSE 和两轮上下文。`eval_agent.py` 输出九条样例的预期/实际工具、FAQ 命中数和原始答案，工具选择通过率不等于回答质量通过率。`demo_agent.py` 展示物流、退货政策和“邮费”字面漏召回的完整 JSON 轨迹。

可用 `--base-url http://127.0.0.1:端口` 指定其他服务地址；两个演示包装脚本也转发该参数。

手动查看接口：

```bash
curl -sN http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"user_id":"demo-user","message":"订单1001的物流到哪了"}'
curl -s http://127.0.0.1:8000/api/agent -H 'Content-Type: application/json' \
  -d '{"user_id":"demo-user","message":"退货政策是什么"}'
curl -s http://127.0.0.1:8000/api/extract -H 'Content-Type: application/json' \
  -d '{"text":"订单 MH20260701123 的猫爬架散架了，我要退款"}'
```

## 离线测试

```bash
uv run --locked pytest -q
uv run --locked python scripts/check_chat_socket.py
```

Windows PowerShell 对应命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_chat_socket.py
```

`check_chat_socket.py` 使用确定性的假流和真实 localhost Uvicorn socket，检查首个 SSE 帧在生产者释放门闩前抵达，再核对后续帧与 `[DONE]`。这是离线传输集成测试，不能代替真实模型验收。全量 pytest 使用隔离的 `mewhelp_test` 数据库；fixture 会重建其中的测试表，不能将其指向业务数据库。

第 1 章曾使用 `gpt-4o-mini` 完成真实模型验收：五条标注样例的订单号与诉求类型均匹配（5/5），两轮 SSE 对话能在第二轮接住“王小明”和“智能猫砂盆”。这属于历史验收结果；第 2 章当时用 `glm-5.2` 验证 tool calling。`expected_solution` 仍需人工核对，其中“询问后续保养”的样例曾返回“未明确”。密钥不会写入仓库，其他机器仍需自行配置 `.env`。

第 2 章的本机 `glm-5.2` tool-call 冒烟已通过；运行 `python scripts/smoke_toolcall.py` 可重新验证当前上游。完整阶段结论与已知限制见 [开发记录](dev-notes/ch02.md)。

## 第 5 章工作流（本机离线验收）

`/api/chat` 和 `/api/agent` 现在共用 LangGraph 会话与 SQLite checkpoint。意图分类把对话送往业务只读工具、强制知识检索、投诉动作或闲聊出口；`MAX_AGENT_STEPS` 默认 6。投诉答复只建议“转人工”和“建工单”，前者是官方渠道指引演示，后者必须由用户填写并确认才会调用 `POST /api/actions/create-ticket`。客户端提供唯一 `request_id`，相同内容重试返回同一工单号；建单不会自动转人工。服务按单 worker 运行，`GRAPH_CHECKPOINT_PATH` 必须指向可写的本地 SQLite 文件。

已有数据库先运行一次非破坏性迁移，再启动服务：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_ch05.py
.\.venv\Scripts\python.exe -X utf8 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

离线验证不会调用聊天或嵌入上游：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\eval_intent.py
.\.venv\Scripts\python.exe -X utf8 scripts\eval_ch05.py
.\.venv\Scripts\python.exe -X utf8 scripts\demo_ui_ch05.py
```

最后一个命令只在 `127.0.0.1:8767` 提供假 SSE 与内存工单，用于浏览器查看按钮交互，绝不代表真实建单。真实服务的流式命令仍用前文 `curl -sN /api/chat`；投诉时 SSE 会额外发送 `actions` 帧。第 5 章 [离线报告](data/ch05/reports/offline_eval.json) 的五路径均为 `passed_offline`，真实 glm-5.2 分类准确率、聊天 SSE 和 JSON 验收仍为 `pending_upstream`，因为账户上游已返回余额不足；未自动重试收费调用。开发过程见 [第 5 章记录](dev-notes/ch05.md)。

服务以 MySQL 消息审计为准。若 SQLite checkpoint 缺失、过期或上轮执行未完成，续聊会返回 503 并在服务日志记录会话号和审计标记；请保留两份数据供排查，用户可开始新对话。当前没有自动恢复旧会话历史的功能。第 2 章的 `eval-agent`、`eval-prompts` 和旧演示脚本使用单轮工具合同，属于历史验收，不适用于第 5 章的新图接口；使用 `eval-ch05` 查看当前离线结论。

第 7 章的 `log/app.log` 记录每轮 `history_ctx` 和每次模型调用的实际 `model_ctx`，包含对话原文，供本机排查。该目录被 Git 忽略；只在受信任的本机账户下运行服务，并定期按运维保留期限清理此文件。浏览器和聊天 API 不提供这些调试日志。

## 第 7 章三层上下文与多会话（离线验收）

已有数据库先运行增量迁移；可重复运行，旧原文与 checkpoint 保留。演示前确认 `DATABASE_URL` 指向需要演示的库。当前本机端口 3307 是隔离测试 MySQL，原 Docker 业务卷尚未恢复，不能把它当作旧业务库。页面左侧可选择旧会话并逐页回载原文；“新对话”保留旧会话。退款选择只在所属会话恢复。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_ch07.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_ch07.py
.\.venv\Scripts\python.exe -X utf8 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

离线评估不连接聊天上游，可用以下占位配置在隔离测试库运行；全量 pytest 还需要知识库测试所用的 Milvus 服务。`eval_ch07.py` 验证默认配置的二十轮原文实际进入模型输入；演示配置的 22 轮案例只验证已完成级联后的模型渲染，另用隔离 MySQL 与假摘要模型测试降级、追加、失败重试和并发去重。标注摘要案例只核对参考答案，输出 [离线报告](data/ch07/reports/offline_eval.json)。真实 glm 二十轮、早期订单追问、摘要质量与 usage 对照仍列为 `pending_upstream`，余额恢复后须在演示库单独验收。

```powershell
$env:CHAT_MODEL='offline-test'
$env:CHAT_BASE_URL='http://127.0.0.1:9/v1'
$env:CHAT_API_KEY='offline-test'
$env:DATABASE_URL='mysql+asyncmy://root:root@127.0.0.1:3307/mewhelp?charset=utf8mb4'
$env:TEST_DATABASE_URL='mysql+asyncmy://root:root@127.0.0.1:3307/mewhelp_test?charset=utf8mb4'
.\.venv\Scripts\python.exe -X utf8 scripts\eval_ch07.py --offline
.\.venv\Scripts\python.exe -m pytest -q tests\test_ch07_budget.py
$env:TOKEN_BUDGET='32768' # 旧 Graph 测试的显式大窗口；预算默认值测试已单独运行
.\.venv\Scripts\python.exe -m pytest -q tests\test_ch07_layers.py tests\test_ch07_summarizer.py tests\test_ch07_repository.py tests\test_ch07_conversations_api.py tests\graph\test_ch07_context.py
Remove-Item Env:TOKEN_BUDGET
```

`SUMMARY_INJECTION_MAX_TOKENS` 默认 250，只注入预算内最新完整摘要段；较早段仍在数据库中，`model_ctx` 会记录未注入段数。`log/app.log` 的 `history_ctx` 和 `model_ctx` 含原文及可能的联系方式，仅限受信任的本机账户读取。按业务保留期限备份必要审计后，停止服务再清理：`Remove-Item -LiteralPath .\log\app.log`。不要将日志或离线占位密钥提交到仓库。

## 第 6 章退款申请流程（样例订单）

先在已有 MySQL 上执行非破坏性迁移，再显式导入固定 `demo-user` 的两条样例订单。重复迁移和重复导入都安全；启动应用不会自动导入演示数据。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_ch06.py
.\.venv\Scripts\python.exe -X utf8 scripts\seed_ch06_demo.py
.\.venv\Scripts\python.exe -X utf8 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

浏览器打开 `http://127.0.0.1:8000/`，点击“使用样例订单演示”（手机端为“样例订单”），再询问“能退吗”。系统仅列出当前演示用户的订单；选择后才从同一会话续跑。政策证据足够时可出现退款申请表；用户选固定原因并点击“确认提交申请”后，独立接口才会生成“待人工审核”申请。它不会执行支付退款或修改订单状态。页面 `user_id` 是演示身份，尚无正式登录认证。

PowerShell 可用以下命令观察 SSE 中断与续跑。先从首个 `interrupt` 帧抄出实际 `conversation_id`，再替换下方的 `123`；恢复流也应以 `done` 和 `[DONE]` 结束。需有可用聊天上游才能完成真实回答，本章不自动触发收费验收。

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' -d '{"user_id":"demo-user","message":"能退吗"}'
curl.exe -N -X POST http://127.0.0.1:8000/api/actions/resume -H 'Content-Type: application/json' -d '{"user_id":"demo-user","conversation_id":123,"order_id":"1001"}'
```

确认写入接口示例（仅用于样例库；将会真正新增待审核申请）：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/actions/create-refund -H 'Content-Type: application/json' -d '{"user_id":"demo-user","conversation_id":123,"order_id":"1001","reason":"质量问题","request_id":"demo-refund-1"}'
```

无模型验收使用隔离 `mewhelp_test` 数据库；不要将测试库配置指向业务库：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\eval_ch06.py
.\.venv\Scripts\python.exe -m pytest -q tests\graph tests\test_ch06_coref.py tests\test_ch06_repository.py tests\test_ch06_resume_api.py tests\test_ch06_refund_api.py tests\test_ch05_actions.py tests\test_ch05_agent_api.py tests\test_ch05_chat_stream.py tests\test_tools_infra.py tests\test_agent_orchestration.py tests\test_agent_stream.py
```

离线页面点击演示：一个终端运行 `.\.venv\Scripts\python.exe -m uvicorn scripts.demo_ui_ch06:app --host 127.0.0.1 --port 8766`，另一个终端运行 `.\.venv\Scripts\python.exe -X utf8 scripts\verify_ui_ch06.py`。该脚本使用本机 Chrome 无头模式和假 SSE，不访问付费模型，也不触碰应用数据库。四路径[离线报告](data/ch06/reports/offline_eval.json)为 `passed_offline`；指代改写与政策扩写的真实效果、真实模型 SSE 仍为 `pending_upstream`，因为此前 glm-5.2 上游返回余额不足且用户选择暂不充值。详见[第 6 章开发记录](dev-notes/ch06.md)。

本机最终回归使用独立端口 `3307` 的临时 MySQL 容器和 `mewhelp_test` 测试库，22 个测试文件分别启动进程，共 113 项通过；Windows 下将这些文件放在单一进程运行曾发生原生扩展异常退出。当前 Docker Desktop 指向空的新数据区，原应用 MySQL 容器未挂载；E 盘 Docker VHDX 备份未改动。因此真实服务演示命令须待原业务数据恢复或在明确的演示库完成迁移和种子导入后运行，不能将临时测试容器当作原业务库。

## 第 3 章当前可用的演示入口

`.env` 还需本机配置 `SILICONFLOW_API_KEY`。真实 SiliconFlow `BAAI/bge-m3` 已返回两条 1024 维向量；`/kb`、`/admin` 页面和相关 API 已接入。Windows 可在项目根目录运行：

```powershell
& 'C:\msys64\usr\bin\make.exe' kb-preview
& 'C:\msys64\usr\bin\make.exe' eval-mining
.\.venv\Scripts\python.exe -m pytest tests\test_kb_manual.py tests\test_kb_api.py -q
```

`kb-preview` 只读源文件，当前输出 11 个知识块；`eval-mining` 会调用已配置的真实聊天上游，输入是仓库内的合成标注样例。浏览器 `/kb` 已实测源文件建库 11 块、补向量、Dense 检索和聊天调用，以及手工录入新增 2 块、重复跳过 2 块；隔离合成库还完成了经确认清库、重建和临时错误密钥造成的向量化失败与重试，四段合成对话挖掘得到 2 条保留、1 条丢弃，最终知识 15 块、pending 0、Milvus 15。网页的补向量与重置在服务进程内运行，避免 Milvus Lite 文件被两个进程同时占用；`eval-retrieval`、`kb-vectorize`、`kb-reset` 命令行目标仅应在 Web 服务停止后运行。重置只清除第 3 章知识块、抽取暂存及 Milvus 知识集合，要求显式确认；应用库未执行重置。Docker/MySQL 恢复后完整测试 **347 passed**、标注检索 `make eval-retrieval` **5/5 passed**；截图和过程见 [第 3 章开发记录](dev-notes/ch03.md)。
