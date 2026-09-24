# PetShop_Helper：电商智能客服（Ch01 纯对话）

FastAPI 服务通过 OpenAI 兼容的 **Chat Completions** 接口连接模型。`/api/chat` 返回 SSE 流并在同一 `session_id` 下保存成功的对话轮次；`/api/extract` 将售后描述提取为订单号、诉求类型和期望方案。本章没有订单查询、人工转接或售后操作工具。

## 配置与启动

需要 Python 3.12、[uv](https://docs.astral.sh/uv/) 和可用的 OpenAI 兼容模型服务。复制 `.env.example` 为 `.env`，填写真实的 `CHAT_BASE_URL`、`CHAT_MODEL`、`CHAT_API_KEY`。示例中的 `YOUR_MODEL_NAME` 和 `sk-your-api-key` 都是占位值。`CHAT_BASE_URL` 应指向供应商的 `/v1` 类兼容接口基址；模型须支持 Chat Completions 流式输出和所选结构化输出方式。默认 `STRUCTURED_OUTPUT_METHOD=json_mode`，若供应商支持也可设为 `json_schema`。请勿把密钥提交到 Git。

Bash（第一个终端）：

```bash
cp .env.example .env
# 编辑 .env，填写真实配置
./scripts/dev.sh
```

Windows PowerShell（第一个终端）：

```powershell
Copy-Item .env.example .env
# 编辑 .env，填写真实配置
.\scripts\dev.ps1
```

服务监听 `http://127.0.0.1:8000`；`GET /health` 应返回 `{"status":"ok"}`。`make dev` 也是启动入口。

## 验收

在服务运行时，于第二个终端执行。脚本会先检查 `.env` 是否存在且已替换占位值；没有真实配置时会明确退出，不会调用上游模型或编造结果。

| 项目 | Bash | Windows PowerShell |
|---|---|---|
| 五条标注提取样例 | `uv run --locked python scripts/eval_extract.py` 或 `make eval` | `.\.venv\Scripts\python.exe -X utf8 scripts\eval_extract.py` |
| 两轮流式对话 | `./scripts/demo_chat.sh` | `.\scripts\demo_chat.ps1` |
| 客服与提取行为样例 | `uv run --locked python scripts/eval_prompts.py` | `.\.venv\Scripts\python.exe -X utf8 scripts\eval_prompts.py` |

提取样例来自 `tests/data/extract_samples.json`。`eval_extract.py` 自动比较 `order_id` 和 `request_type`，打印逐条结果与总通过数；`expected_solution` 需要人工核对是否忠实、合理。`demo_chat.py` 验证 SSE 协议完整，并用同一会话询问名字和商品；请人工核对第二轮是否正确复述“王小明”和“智能猫砂盆”。`eval_prompts.py` 根据 `tests/data/prompt_cases.json` 输出模型回答与逐条人工复核清单，覆盖虚构订单状态、保证退款时效、声称已转人工和提示词注入；它只报告请求/协议错误，不自动声称语义质量合格。

可用 `--base-url http://127.0.0.1:端口` 指定其他服务地址；两个演示包装脚本也转发该参数。

手动查看接口：

```bash
curl -sN http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s1","message":"你们卖猫粮吗？"}'
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

`check_chat_socket.py` 使用确定性的假模型和真实 localhost Uvicorn socket，检查首个 SSE 帧在假模型释放门闩前抵达，再核对第二帧与 `[DONE]`。这是**离线传输集成测试**，没有连接真实模型，不能代替上述真实模型验收。

当前仓库没有已配置的 `.env`，因此尚未运行真实上游的五条提取、两轮上下文和行为样例评估；这些结果须在配置模型后执行并人工复核。
