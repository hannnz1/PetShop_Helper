# 第 8 章工具系统演示

本章工具的订单、物流、售后和优惠数据均为模拟演示数据。运行目录是第 8 章工作树。正式模型验收依赖 `.env` 中可用的聊天模型额度；离线检查不调用上游。

## 启动

在 PowerShell 中切到本目录。先确认应用 MySQL schema 已执行 `sql/ch08-ddl.sql`，再启动：

```powershell
make mcp-up
make dev
```

`make dev` 启动主应用并确保两台本地 MCP 服务在 8101/8102；浏览器打开 `http://127.0.0.1:8000/`。结束时运行 `make mcp-down`。独立离线页面检查可用 `..\..\.venv\Scripts\python.exe -m uvicorn scripts.demo_ui_ch08:app --host 127.0.0.1 --port 18888`，只返回合成 SSE，不写数据库。

## 六条验收路径

1. 新内置工具：`Copy-Item scripts/demo_ch08_promotions.py.txt app/tools/builtin/promotions.py`，重启主应用，问“最近有什么优惠活动”。应显示 `query_promotions` 调用且答复标明模拟演示。演示完用 `Remove-Item -LiteralPath app/tools/builtin/promotions.py` 并重启。仓库默认不安装这个演示工具。
2. 双服务发现：问“我的订单 1001 的物流到哪了”和“订单 1001 还在保吗”。物流应先调用 `query_order` 再调用 MCP `query_logistics`；在保查询由售后 MCP 返回模拟状态。
3. 动态 MCP：在 `mcp_servers/logistics_server.py` 临时登记一个只读工具，运行 `make mcp-down`、`make mcp-up`，不重启主应用；新工具应出现在下一轮业务对话可绑定清单。演示后还原文件并重启 MCP。
4. 确认建单：问“帮我建个工单”，缺少问题时应追问；补充“猫砂盆漏电”后出现预览卡，点击“确认提交”，回复应含工单号。`tickets` 应只增加一条，`tool_audit_logs` 应记录 `create_ticket` 成功。
5. 取消建单：另开一轮走到预览卡，点击“取消”。`tickets` 不增加，`tool_audit_logs` 中该调用状态应为“权限拒绝”。
6. 故障路径：使用 `MOCK_DELAY_SECONDS=12` 单独重启物流 MCP 服务后再问物流，应在工具超时后给出可理解的失败答复，审计应记录“超时”和只读重试次数；恢复正常服务。建单写操作的超时验收另在隔离库验证，不会自动重试。

## 评估和测试

```powershell
make eval-ch08-offline
make eval-ch08
make test
```

`eval-ch08-offline` 只校验五条标注样例，状态 `pending_upstream` 表示真实模型尚未验收。`eval-ch08` 会向运行中的应用发送真实模型请求，执行确认和取消场景并核对数据库记录；只有 `.env` 模型额度与服务都可用时才运行。测试必须指向独立 `*_test` schema，切勿将测试 URL 指向正式应用库。
