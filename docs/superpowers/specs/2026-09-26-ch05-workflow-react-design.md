# PetShop_Helper 第 5 章设计：Workflow 骨架与有界 ReAct

> 状态：书面 spec 已获用户批准，实施计划待评审。权威来源：`C:/Users/Administrator/Documents/PetShop_Helper开发文档/5/理论学习：Workflow 编排与 ReAct 循环.md`。Desktop 旧 MewHelp 的 Ch05 文档仅供比较，不继承其历史决策。

## 目标与边界

现有 `app/core/agent.py` 每轮最多先规划一次工具、再生成答复，不能在工具结果出现后决定下一步。第 5 章将两个聊天入口统一接入 LangGraph：固定路由和证据闸由 Workflow 管，业务工具组合由步数受限的 ReAct 环处理。现有 FastAPI、LangChain、MySQL、Milvus、SiliconFlow、OpenAI 兼容上游保持原选型。

第 4 章正式 300 题生成评估因 glm-5.2 余额不足仍为 `partial`；本章离线开发不改变这项结论，也不触发付费模型调用。指代消解的正式实现、复杂意图、上下文分层、MCP 与可观测性/飞轮深化分别留到后续章节。

## 流程

```text
START → coref(本章透传) → classify_intent(七类单标签)
  ├ 闲聊 → 固定话术 → log → END
  ├ 投诉 → 安抚话术 + 可选动作 → log → END
  ├ 商品咨询/退款退货 → forced_rag → 证据闸
  │                              ├ 弱 → 拒答 + 低置信池 → log → END
  │                              └ 强 → agent_llm ↔ agent_tools → log → END
  └ 物流/订单/其他售后 → agent_llm ↔ agent_tools → log → END
```

知识路径必须在生成前拿到第 4 章的编号证据和引用，沿用现有检索/自评规则；模型不得绕过强制检索重新凭空回答。图内把检索与自评统一为 `sufficient/evidence/citations/reason` 结果；当前本地 Milvus 分支 `query_faq` 只返回 `hits`，不得把缺少 `sufficient` 当作强证据，未规范化或异常时一律拒答并留痕。业务路径不使用知识证据闸。投诉和闲聊不进入 Agent；投诉只提出“转人工”“建工单”，不自动调用写入工具。

七类意图到四出口固定映射：物流、订单、售后→business；商品咨询、退款退货→knowledge；投诉→complaint；闲聊→chitchat。意图分类输出错误、无标签或上游不可用时，按可审计的兜底路径结束，不能静默转入可执行工具的业务路。意图分类 Prompt 用标注样例验证，不把单元测试伪装成模型效果验证。

## 有界 ReAct 与现有工具

`agent_llm` 根据当前状态决定工具调用或收敛，`agent_tools` 只通过现有 `app/tools/infra.py` 执行白名单工具，并把结果返回给下一次 `agent_llm`。模型无工具调用后进入独立的 `final_answer` 节点，用禁用工具的模型调用逐 token 生成面向用户的答复；中间分类、检索自评、工具规划输出绝不发到 SSE。达到 `max_agent_steps`（默认 6）、消息预算不足或工具基础设施失败时输出明确兜底，不把工具结果原文冒充最终答复。每次模型响应的 token 用量可累加到 trace，但不得以该数字代替已存在的输入预算检查。

现有四项只读工具中，业务路径仅可绑定 `query_order/query_product/query_logistics`；知识路径强制预检索后仅可绑定 `query_order` 以查询相关订单，政策/商品知识结论必须由已通过证据闸的引用支持，不能用模拟商品价格或再次调用 FAQ 绕过证据。当前 `query_logistics(order_id)` 不因 Desktop 旧方案改成 `tracking_no`；旧项目“先手机号查订单再查运单”的示例不是当前工具能力。本章用可控的多步假模型证明图能在先前工具结果后继续决策，真实业务依赖若需新工具/参数，另行设计并评审。

`create_ticket` 从聊天模型绑定工具清单移除，可信执行分发层也拒绝任何模型发来的同名调用；原 `AGENT_SYSTEM` 中“直接创建工单”和“每轮只调一次工具”的指令同步改掉。投诉路径由确定性节点产生建工单/转人工选项；其他路径不自行建单。真正创建只由用户按钮发起。读工具超时/失败留在工具轨迹中供模型判断是否追问，写操作不能自动重试。

## 状态、持久化与接口

图状态包含消息历史（`add_messages` 归并）、`user_id`、`conversation_id`、当前问题、意图/路由、证据与引用、最终答复、工具轨迹、步骤数、token 用量、建议动作及可审计 trace。每轮开始重置只属于当轮的证据/动作/步骤，保留跨轮消息。MySQL 继续作为会话归属、消息审计、工单和低置信问题的业务真相；`AsyncSqliteSaver` 用独立文件和 `thread_id=str(conversation_id)` 持久化图执行状态。先验证用户对会话的归属，再读取或写入对应 thread，禁止用提交的 `thread_id` 绕过归属检查。单进程部署内按 conversation_id 串行处理轮次；同会话并发请求返回 409，避免交错写检查点和审计消息。第 5 章运行约束为单 worker，多 worker 跨进程锁留后续独立设计，不能声称已支持。SQLite 与 MySQL 的两份记录发生不一致时，以 MySQL 会话/消息为对外依据，记录可恢复错误，不能泄露其他用户的检查点。

`/api/agent` 使用图的非流式执行返回原有 answer/tool_calls/tool_results，并扩展 `suggested_actions`；`/api/chat` 用图的消息与节点更新流保持逐 token SSE、工具事件、引用事件、done 事件，新增 `actions` 事件。消息流严格按节点元数据过滤，仅 `final_answer` 的面向用户文本可作 `delta`；工具调用片段和内部分类/自评/规划 token 均不对外推送，固定话术由对应出口单独发出。两个入口共享同一图与同一会话状态，不保留并行的旧编排入口。应用 lifespan 负责创建与关闭 checkpointer；离线测试使用隔离 SQLite 文件。

动作接口 `POST /api/actions/create-ticket` 接收当前 `user_id`、`conversation_id`、工单类型、描述及客户端生成的 `request_id`；先校验会话归属及必填内容，再写工单并返回单号。`tickets.request_id` 唯一，重试同一请求返回原单号，参数不同复用同一 ID 返回冲突。**现有** `repository.create_ticket` 同时把会话状态改为“已转人工”，与本章两个独立按钮的语义冲突；实施时要拆出“仅建工单”的仓储操作，按钮端点不得隐式转人工。转人工仅是前端模拟，不声称接入真人系统。聊天页显示两个独立按钮；建工单需用户填写并确认，取消不写库，提交成功后该按钮禁用，继续对话不受影响。聊天页面遵照用户指定的 Vibe Coding 方式直接迭代，不套 brainstorm/TDD/code review。

## 验证与验收

1. 无模型冒烟：已安装 LangGraph 版本上的图编译、条件边、持久检查点、非流式调用和消息/更新流形状与 [Context7 核对](../../context7-ch05.md)一致。
2. 确定性代码 TDD：七类路由、弱证据短路、步数停止、工具继续循环、会话归属、同会话并发 409、建单幂等及动作只在点击后写库、SSE 内部 token 不泄漏与跨轮恢复；测试使用假模型和隔离数据库，不付费。
3. Prompt/数据以七类标注意图集和端到端路径集验证；记录每类正确率、误分和修订，不以 Prompt 文本存在代替效果。真实模型额度恢复后补跑五类路径及多步业务的真实 SSE/JSON 冒烟。
4. 浏览器验收：知识问句引用仍能点开原文，投诉有两个互不绑定的选项，取消建单不写库、确认后有单号；跨轮同一会话可续接，越权会话返回错误。

阶段结论必须区分“离线代码通过”与“真实模型端到端通过”；第 4 章部分报告以及第 5 章真实模型验收不得被离线测试覆盖。

## 风险与决策

- 第 5 章理论文档没有可执行 plan；本 spec 把现有项目接口和后续章节边界落地。正式实施计划需独立评审。
- 旧项目 `bm25_text` 参数与当前 `bm25_query` 不同、旧项目直接导入 `settings` 与当前 `get_settings()` 不同，不能照抄。旧项目建工单端点未保留当前会话归属校验，必须修正。
- 当前 `create_ticket` 模型工具可直接写库，且底层仓储隐式更改会话状态；新图必须同时移除绑定并在可信工具分发层拒绝模型的写入调用，按钮端点使用不转人工的写入路径。与第 2 章旧测试涉及的行为差异在实施计划中显式迁移。
- 当前 `pyproject.toml` 尚无 `langgraph`、`langgraph-checkpoint-sqlite`、`aiosqlite`；实施计划的第一阶段要锁定依赖并用本地无模型冒烟验证，不能只依据文档推定兼容。
- 上游余额不足期间只做离线验证；真实模型验收和第 4 章 726 条生成失败的补跑均保留待办，不改用另一服务商冒充通过。
- LangGraph 具体接口已通过 Context7 查到官方形状，仍以首个本机 fail-fast 冒烟确定安装版本行为。
