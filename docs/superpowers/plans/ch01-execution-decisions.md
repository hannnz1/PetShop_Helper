# Ch01 实施裁定与任务状态

原计划：2026-09-23-ch01-pure-chat-plan.md。用户提供 Spec/Plan 后要求“继续完成开发”，视为授权按该设计实施。本文件澄清原稿矛盾，优先于原稿示例；不更换技术栈。

## 计划评审与接口裁定

| 任务/依赖 | 检查结果与裁定 |
|---|---|
| 1 配置 | 保留 Settings 与 get_settings() 缓存工厂，应用启动校验；取消 import 时强制实例化以使无凭据单测可收集。CHAT_API_KEY 使用 SecretStr；仅三项 CHAT 必填。 |
| 2 schema | 保留接口字段和枚举；拒绝空白字符串，order_id 缺失 null。 |
| 3 memory | SessionStore 按会话保存完整成功轮次；预算默认 2000，包含 system+历史+本轮输入的估算 token。过长本轮返回 422，绝不悄悄删掉当前输入。输出上限独立 CHAT_MAX_TOKENS=1024，上游窗口需同时容纳二者。 |
| 4 prompt | ChatPromptTemplate；纯提示词效果用评估集，不用关键词测试假装效果验证。无工具，不谎称已经查询或完成转人工。 |
| 5 LLM | ChatOpenAI 固定 use_responses_api=False、stream_usage=False，base_url/model/api_key 从配置读取。显式 method=json_mode 默认，可配置 json_schema；不自动回退到 function_calling。 |
| 6 SSE | data JSON delta + 空行，成功 [DONE]，失败 event:error 后关流；失败不入历史；同会话并发返回 409。先消费上游内容即转发，不人工拆字冒充模型 token。 |
| 7 extract | with_structured_output(AfterSalesTicket, method=配置)，502 对外通用错误；默认 JSON 模式 prompt 包含完整字段约束。 |
| 8 验收 | 离线 pytest + 实际网络 SSE 传输测试；真实 .env 配置后完成五条提取、多轮上下文和客服行为样例评估。缺少真实密钥时保留未完成状态，不能用 mock 充当真实验收。 |
| 1→5/6/7 | get_settings() 返回 Settings，依赖注入覆盖隔离真实凭据；所有超时/输出上限正数。 |
| 2→6/7 | 请求严格非空；响应只包含既定字段与五种枚举。 |
| 3+4→6 | 先生成 system，再预留本轮输入预算，仅裁剪旧的完整轮次。成功存储裁剪后的完整历史，防止单会话无限累积。 |
| 4+5→7 | JSON 模式提示必须明确 JSON 格式；无 Agent、无工具调用。 |
| 5→6/7 | 超时不返回内部错误或密钥；不硬编码供应商型号。 |
| 6+7→8 | 相同接口契约，评估脚本遇到 SSE 错误或缺少 DONE 即失败。 |

## 执行说明

- Python 3.12 + uv + FastAPI + LangChain 保持固定；Windows 提供 scripts/dev.ps1，并保留 Bash 入口。
- 用户询问 OpenAI Key 可用性，按 OpenAI 兼容协议支持；真实验收以用户最终保存的 .env 为准，不自行使用旧稿端点。
- 本章仅使用 CHAT 配置，EMBED/RERANK 留后续章节。
- Context7 原生工具尚未刷新，通过已验证的官方 MCP HTTP tools/call 读取文档，仍是 Context7 MCP，不以模型记忆替代。检索证据记录在 docs/context7-ch01.md。
- 当前是新建且无 Git 历史的独立目录，在功能分支初始化，不涉及已有主分支或用户修改；无需额外复制空仓库到 worktree。
- 提交署名采用真实本地 Git 身份，不伪造原稿的其他助手署名。

## 状态

- [x] Task 1 配置
- [x] Task 2 Schema
- [x] Task 3 Memory
- [x] Task 4 Prompt
- [x] Task 5 LLM 与启动
- [x] Task 6 SSE
- [x] Task 7 结构化提取
- [x] Task 8 验收工具与离线验证

真实模型验收待 `.env` 配置后执行；全量离线测试 80 项通过，真实 localhost socket 的假模型时序测试通过。
