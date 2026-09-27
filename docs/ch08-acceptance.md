# 第 8 章验收记录

日期：2026-09-27。实现分支：`feat/ch08-tool-system`。本记录区分代码与离线验证、真实服务验证及仍待上游恢复的端到端验收；`pending_upstream` 不是 PASS。

| 计划验收路径 | 已验证的结果 | 尚待验证 |
| --- | --- | --- |
| 新增内置促销工具 | 临时复制演示模块后，启动扫描自动登记；活动意图可选用新只读工具；演示文件已移除。 | 有额度的真实模型对话中是否选择并正确描述活动。 |
| 订单物流与售后 MCP | 两台真实本地 HTTP MCP Server 可发现工具；Graph 可把物流调用交给统一引擎；中文状态格式化和审计有隔离测试。 | 真实模型连续调用订单、物流以及售后工具的完整会话。 |
| 仅重启 MCP 后发现新工具 | 实际本地 MCP 服务重启测试通过，主应用未重启，下一轮工具清单出现新工具。 | 真实模型是否按问题选中新工具。 |
| 建单预览并确认 | Graph 中断/恢复、HTTP 返回帧、MySQL 隔离库持久化及离线页面点击均已验证。确认写入只执行一次并带工单号。 | 配置真实模型与应用业务库后的浏览器全链路。 |
| 预览后取消 | Graph/HTTP 和离线页面取消路径已验证，无工单新增，审计记录权限拒绝。 | 真实模型及业务库下的浏览器全链路。 |
| 工具超时与重试 | 真实延迟 MCP 请求验证只读超时重试；写工具模拟延迟验证不自动重试；审计状态与次数有断言。 | 运行中业务应用接真实延迟服务的完整浏览器场景。 |

五条标注场景可用 `make eval-ch08-offline` 校验，当前全部标记为 `pending_upstream`。真实 `make eval-ch08` 需可用的 glm-5.2 额度；此前上游返回 HTTP 429 / 1113 余额不足，用户选择暂不充值，所以未调用付费模型进行这五题验收。测试运行于独立 MySQL `mewhelp_test`，正式应用 `mewhelp` schema 未在该测试容器配置，未做正式库迁移。

离线集中回归为 **150 passed**；广域离线套件复跑为 **596 passed、1 deselected、1 warning**。广域命令在独立测试库上设置 `TOKEN_BUDGET=32768`，因此单独排除了断言默认预算 2000 的配置测试；该测试去掉覆盖变量后单独 **1 passed**。广域命令还排除了需当前不可用 Milvus 或正式业务库的 `test_dualwrite_ch04.py`、`test_milvus_hybrid.py`、`test_retrieval_ch04_live.py`、`test_mining.py`。前一次运行的 2 秒并发等待偶发超时已单独复现为通过，并将测试等待上限放宽至 10 秒；复跑全绿。`compileall` 与 `git diff --check` 通过。独立后端 code review 的五项意见已修复并经只读复核，无剩余严重或重要问题；页面按用户规定采用 Vibe Coding，不纳入该 review。

演示操作与复现命令见 [ch08-demo.md](ch08-demo.md)。实现方案见 [spec](superpowers/specs/2026-09-27-ch08-tool-system-design.md) 与 [plan](superpowers/plans/2026-09-27-ch08-tool-system-plan.md)；开发过程见 [ch08.md](../dev-notes/ch08.md)。
