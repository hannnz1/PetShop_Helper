# PetShop_Helper：电商智能客服（第 1–2 章）

FastAPI 服务通过 OpenAI 兼容的 **Chat Completions** 接口连接模型。`/api/chat` 返回带工具轨迹的 SSE 流，使用 MySQL 保存多轮会话；`/api/agent` 返回完整 JSON 及工具轨迹；`/api/extract` 将售后描述提取为固定字段。第 2 章包括五个 LangChain 工具：三个模拟数据查询、MySQL FAQ 查询和工单创建。订单、商品、物流查询结果均为演示数据。

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

第 1 章曾使用 `gpt-4o-mini` 完成真实模型验收：五条标注样例的订单号与诉求类型均匹配（5/5），两轮 SSE 对话能在第二轮接住“王小明”和“智能猫砂盆”。这属于历史验收结果；当前本机 `.env` 已切换至第 2 章 `glm-5.2`。`expected_solution` 仍需人工核对，其中“询问后续保养”的样例曾返回“未明确”。密钥不会写入仓库，其他机器仍需自行配置 `.env`。

第 2 章的本机 `glm-5.2` tool-call 冒烟已通过；运行 `python scripts/smoke_toolcall.py` 可重新验证当前上游。完整阶段结论与已知限制见 [开发记录](dev-notes/ch02.md)。
