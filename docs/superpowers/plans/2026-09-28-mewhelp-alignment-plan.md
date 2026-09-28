# MewHelp 功能对齐 Implementation Plan（总索引）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成已批准补充 spec 的 21 项功能对齐，并以本项目最终版本完成 1–10 章真实验收。

**Architecture:** 在现有 worktree 复用应用工厂、Graph、MySQL、审核服务与轻量分类器。分四个可验证的子计划，接口与报告沿用统一版本标识，保持现有权限、隐私和幂等保护。

**Tech Stack:** Python 3.12、FastAPI、LangChain/LangGraph、SQLAlchemy/MySQL、Milvus、Langfuse、PyTorch/ONNX、静态 HTML/JS、Windows PowerShell。

**Spec:** [已批准的补充设计](../specs/2026-09-28-mewhelp-feature-alignment-design.md)。用户于 2026-09-28 回复“批准补充 spec”和“批准实施计划”，进入连续实施。

## Global Constraints

- 根目录为 `C:/Users/Administrator/Documents/Codex/2026-09-23/kai/PetShop_Helper/.worktrees/ch08-tool-system`，产品基线 `abe481e`；实施前记录当时 HEAD，不新建重复 worktree，不覆盖 `.env`。
- 固定技术栈、上游协议和模型保持；遇到选型冲突停下来向用户说明。具体 API 实现前先查 Context7 官方文档，再核对 uv.lock/安装版本，沿用已有有效查询记录。
- 每批含实现、验证和留痕；已批准范围不再重复 brainstorm。采用此前用户偏好的连续实施、阶段集成检查、最终独立复核；用户已批准执行本计划。
- 后端 TDD；Prompt/数据以标注评估替代 TDD；聊天页沿用 Vibe Coding＋浏览器验收，其余后台页按模块正常复核。
- 禁止以 fake runtime、partial、pending 或参考历史报告冒充真实验收；未完成真实项时总清单不打勾。
- 每项完成当时追加 `dev-notes/mewhelp-alignment.md` 四字段记录，并关联原章节。任务提交仅包含该任务文件；Git 身份缺失时保留改动并记原因，不伪造作者或改全局配置。
- 已批准的 300 题批次成功缓存按版本复用；第 10 章真实文本外发不借用第 4 章授权。缺预算/数据/算力不阻止其它离线代码工作，但不能降低真实验收标准。

## Review Focus

1. 切换会话后点旧气泡，不得反馈给新会话：I-2 浏览器与消息 ID 测试。
2. 从旧 `/api/jobs` 直接启动新作业，不得绕过页面鉴权：II-1 的所有入口矩阵测试。
3. 分类服务热更新导致同一批两个模型版本，不得混写：III-4/III-5 元数据与版本不符测试。
4. 有旧报告但对应数据/模型已变化，不得在后台显示当前 PASS：III-7 版本失配测试。
5. 收费批次中途失败，续跑不能把成功项重跑或把失败项当零分：IV-1 缓存指纹及失败分母核验。

## 子计划与依赖

| 顺序 | 计划 | 可独立验收的交付 |
|---|---|---|
| I | [基线、反馈与检索](2026-09-28-mewhelp-alignment-core-plan.md) | 正确的反馈入池、校准与检索质量行为 |
| II | [运营后台](2026-09-28-mewhelp-alignment-operations-plan.md) | 受保护的作业、观测与审核操作 |
| III | [分类器与主题闭环](2026-09-28-mewhelp-alignment-classifier-plan.md) | 语料补全、分类落库、主题与验收后台 |
| IV | [真实验收与交付](2026-09-28-mewhelp-alignment-acceptance-plan.md) | 真实报告、最终入口及全章交付 |

I 完成后执行 II；III 的纯数据任务可提前，但默认连续执行节省协调成本。II-1 先提供权限分发；只有 III 实际脚本就绪后才注册相应作业。IV 以各批阶段报告为输入，不重复实现已通过能力。

## 清单逐项映射

| 清单 ID | 拥有任务 | 验收完成才关闭 |
|---|---|---|
| A01 | I-1、IV-3 | 最新入口和集成版本可复演 |
| A02 | I-1、IV-3 | spec/plan/报告与清单状态一致 |
| F01 | I-2、IV-1 | 点击气泡到真实隔离 MySQL |
| R01 | I-3、I-4、IV-1 | 计算/集成/真实校准分别有证据 |
| R02 | I-4 | 错 category 回退且硬约束保留 |
| R03 | I-5 | 评估型号校验与修复受限 |
| R04 | IV-1 | 300题×四策略完整且质量结论明确 |
| O01 | II-2、II-4 | 页面三个区块数据与状态可信 |
| O02 | II-2、IV-1 | 用量统计及本机 Langfuse 对账 |
| O03 | II-2 | 同版本报告的数字说明 |
| W01 | II-3、II-4 | 页面审核/发布及向量状态 |
| W02 | II-3、IV-1 | 三入口到再次检索命中 |
| C01 | III-1 | 修错样例质量与完整血缘 |
| C02 | III-2 | CLI 真接增强、holdout 不变 |
| C03 | III-3 | P/R/F1、矩阵、错例一致 |
| C04 | III-4、III-5 | 分类元数据与幂等落库 |
| C05 | III-6 | 主题分布/分页与 SQL 一致 |
| C06 | III-4、III-7 | healthz、九项验收及作业保护 |
| C07 | IV-2 | 足量数据、真实训练/导出/服务 |
| D01 | IV-3 | 统一后台、PowerShell 部署 |
| D02 | IV-3 | 逐章演示样例和入口 |

## 测试约定

在独立 PowerShell 测试终端中进入上述 worktree，使用其现有虚拟环境；不要把离线变量写进 `.env`：

```powershell
$env:CHAT_MODEL='offline-test'
$env:CHAT_BASE_URL='http://127.0.0.1:9/v1'
$env:CHAT_API_KEY='offline-test'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
function Test-Alignment {
  & .\.venv\Scripts\python.exe -X utf8 -m pytest -q -p pytest_asyncio.plugin @args
  if ($LASTEXITCODE -ne 0) { throw 'Alignment test failed' }
}
```

各子计划 `Test-Alignment <测试路径>` 的通过标准为 0 failed/0 errors；RED 必须是新增行为的断言失败，不以缺数据库/导入依赖代替 RED。数据库测试单独配置隔离库 `TEST_DATABASE_URL`，禁止指向正式库；不同预算用 fixture 注入，不能任意全局改 TOKEN_BUDGET 让旧断言失真。真实命令使用另一终端加载本地真实配置。

每个任务结束记录命令、样例版本、实际结果和剩余 pending，更新勾选并提交该任务文件。每批只做一次必要的相关模块集成，只有新增失败/修改才扩大或重复。无代码改动的文档任务只校验链接、条目和命令，不为文档编造单测。

## 计划自查

- [x] 21 个清单 ID 全部映射；实现完成与真实验收关闭条件分开。
- [x] 复用已有 audit_message_id；无需新增回答表或以 latest 查询反推 ID。
- [x] 强制 RAG 先做新置信度闸，避免把它错误接在现有工具自检之后；政策路径保持原规则。
- [x] 新作业授权在所有 API 分发入口统一验证，不能仅保护页面。
- [x] 分类结果表、批次、服务模型版本、读图报告版本有对应实现/测试任务。
- [x] 对分类增强中的同步函数和异步模型调用做预计算适配，防止协程当字符串。
- [x] 计划未包含自动付费日程、清库或未授权真实文本外发。

## 执行评审

用户批准本组计划后，以 `superpowers:executing-plans` 连续实施；最终依 `requesting-code-review` 做独立复核。若用户更改执行偏好再调整，不默认逐任务新建 subagent。
