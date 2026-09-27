# 第 9 章第二部分验收：知识数据飞轮

日期：2026-09-27。范围为已批准的 `docs/superpowers/specs/2026-09-27-ch09-knowledge-flywheel-design.md` 和对应计划。全部写入/测试在隔离 MySQL `mewhelp_test`（本机端口 3307）完成；未调用付费聊天模型、SiliconFlow 或真实业务库。

## 可复现命令

在仓库根目录 PowerShell 中先设置不含真实密钥的离线测试变量：

```powershell
$env:CHAT_MODEL='offline-test'
$env:CHAT_BASE_URL='http://127.0.0.1:9/v1'
$env:CHAT_API_KEY='offline-test'
$env:DATABASE_URL='mysql+asyncmy://root:root@127.0.0.1:3307/mewhelp?charset=utf8mb4'
$env:TEST_DATABASE_URL='mysql+asyncmy://root:root@127.0.0.1:3307/mewhelp_test?charset=utf8mb4'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:TOKEN_BUDGET='32768'
.venv/Scripts/python.exe -X utf8 -m pytest -q -p pytest_asyncio.plugin tests/flywheel tests/test_low_confidence.py tests/test_ch07_conversations_api.py tests/graph/test_knowledge_route.py tests/test_admin_api.py tests/test_dualwrite.py
```

整合回归在最终并发修复前得到 **45 passed, 1 个既有 Starlette/AnyIO 弃用告警**；并发修复后针对 `tests/flywheel` 全套重跑 **23 passed**。受保护 API 的合成数据 `approve→publish` 路径、CLI 的 `list --limit 2` 路径均成功。标签集 `tests/flywheel/canonicalization-labels.json` 覆盖同义问法、政策差异、个人信息与非法候选 ID；离线假模型验证了结构、遮蔽、归并及拒绝越界 ID，不把它解释为真实模型的语义准确率。

## 验收结果

| 环节 | 本地证据 | 状态 |
| --- | --- | --- |
| 三种缺口入口 | 现有 `self_check`、`retrieval_low_conf` 测试和新增用户反馈归属/幂等测试 | PASS |
| 标准问句归并 | 两问句归一 ID、旧候选超出 30 条窗口、非法 ID、候选中途关闭、重复批次测试 | PASS（假模型） |
| 审核保护 | token 未配置 503、错误/缺失 401、授权列表及 API 批准测试 | PASS |
| 人工批准和发布 | 人填答案才批准；MySQL 待向量化、双发幂等、插入后中断恢复、冲突 ID 禁止脏发布测试 | PASS |
| 向量化后可检索 | 两项旧 Milvus 集成测试因 `127.0.0.1:19530` 无服务而无法建连；`docker ps -a` 无 Milvus 容器 | `pending_upstream` |
| 真实模型问句质量 | glm-5.2 额度仍不可用，且无真实用户文本外发授权 | `pending_upstream` |

首次加上 `tests/test_dualwrite_ch04.py` 的扩展运行得到 **40 passed、2 errors**；两个 error 均发生在 Milvus 连接 fixture，服务地址 `127.0.0.1:19530` 不可用，不能算代码功能通过。随后在明确排除该服务依赖的范围内得到上述 45 passed。MySQL 待向量化行不等于已检索可用；本报告不把第 4 章完整 300 题或第 8 章真实模型验收计作通过。

## 操作入口和边界

在配置正式库前，先通过 `python -m scripts.ch09.migrate_flywheel` 应用增量表；该脚本目前只在隔离测试库验证，正式业务库尚未运行。反馈为 `POST /api/feedback/unresolved`。审核使用独立环境变量 `KNOWLEDGE_REVIEW_TOKEN` 的 Bearer，路由是 `GET /api/review/questions`、`POST /api/review/questions/{id}/decision` 和 `POST /api/review/questions/{id}/publish`；本机 CLI 为 `python -m scripts.ch09.flywheel_review list|decide|publish`。标准化 CLI `python -m scripts.ch09.flywheel_canonicalize --limit 20` 对非本机上游默认只报 `pending_upstream`，不会外发原始问句。发布后需既有 `vectorize_pending` 作业成功，才会变为可检索。

独立复核发现并已用 RED→GREEN 回归修正：发布前请求 ID 冲突会留下待向量化脏行；旧标准问句唯一键导致原始问题永久卡住；模型调用期间候选关闭仍可挂载；新发布请求未审计；跨问题并发共用审核请求 ID 返回数据库异常。最后一项的并发回归先失败、修复后通过。复核结论是这些 Important 问题已关闭；未发现新的 Important/Critical。
