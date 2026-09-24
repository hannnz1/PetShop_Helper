# Ch02 实施裁定（源 Plan 的约束补充）

源文档：[Ch02 Spec](../specs/2026-09-24-ch02-business-tools-design.md)、[Ch02 Plan](2026-09-24-ch02-business-tools-plan.md)。2026-09-24 独立计划预审提出 4 个阻塞和 4 个高/中风险，以下裁定优先于源 Plan 中的示例代码。技术选型不变。

1. **Task 1 是真实 `glm-5.2` go/no-go 闸。** 冒烟脚本必须检查 `CHAT_MODEL == "glm-5.2"`，并验证配置为提供该模型的上游地址；只在实际响应中出现结构化 `tool_calls` 时报告通过。当前 `.env` 为第 1 章 OpenAI 模型，不能用于此闸。未获得可访问 `glm-5.2` 的配置前，仅做文档/离线准备，不标注通过、不进入依赖模型工具调用能力的实施。密钥和响应原文中的敏感字段不写日志。
2. **配置接口跟随第 1 章实际代码。** 使用 `Settings`/`get_settings()`，不导入不存在的 `app.config.settings`。DB 相关配置新增字段按需注入，不在模块导入时连接数据库；测试可传入替代 Settings。
3. **DDL 与测试库。** `sql/ch02-ddl.sql` 是四表权威建表结构，包含 `SET NAMES utf8mb4`。测试库 fixture 必须正确执行有注释的 SQL 文件，不能靠 `strip().startswith("CREATE TABLE")` 跳过全部建表；只允许在显式的隔离测试库内清理或重建表，不默认删除整个数据库。MySQL 实测使用指定 Docker 实例，不拿 SQLite 冒充。
4. **历史预算。** Ch01 `trim_history()` 只接受完整 Human/AI 轮次。Ch02 持久化消息有 tool 角色和可能的未成对记录，不能直接把 SystemMessage/奇数历史传入。Task 11 必须构造可被模型接受的历史、计入 system 与当前用户预算，并写首轮/续轮/工具结果测试；需要新的裁剪策略时在不改固定语义的前提下单独实现。
5. **系统注入参数。** `create_ticket.conversation_id` 使用 `InjectedToolArg` 且只由服务端可信会话 ID 注入。测试对模型可见 schema（`tool_call_schema`/实际绑定请求）断言该字段缺席，对执行 schema 断言可注入；不可采用源 Plan 中去掉注入标记的 fallback。工具执行前覆盖或拒绝模型自带的同名参数。
6. **工具状态帧时间。** 流式出口必须在等待工具执行之前发送选中工具的 SSE 状态帧；工具完成后再回灌模型生成最终答案。保持共享会话、规划和落库核心，不为两个出口复制工具业务逻辑。
7. **FAQ 漏召回验收。** Seed 使用 `SET NAMES utf8mb4`；直接测试 `keyword="退货政策"` 命中及 `keyword="邮费"` 对“运费怎么算”不命中。浏览器真实模型测试记录工具参数与实际结果；若模型自行改写“邮费”为“运费”，不能把那次命中当作 LIKE 漏召回验收。
8. **映射与错误分层。** ORM 应明确匹配 DDL 的无符号主键及 MySQL 服务器维护的更新时间。工具业务失败可回灌模型；数据库连接/事务等基础设施故障按接口约定处理，不能一律吞为 `ok=false` 的 200 响应。写类 `create_ticket` 不自动重试。

## 当前闸状态

- Ch01 真实 OpenAI 验收进行中；Ch02 `glm-5.2` 上游配置尚无。用户已获 Z.ai 官方充值/API Keys 入口和固定模型要求。
- 此文档是预审问题的裁定稿，须做独立复审后才可把源 Plan 标为可执行；Task 1 冒烟仍需真实凭据。
