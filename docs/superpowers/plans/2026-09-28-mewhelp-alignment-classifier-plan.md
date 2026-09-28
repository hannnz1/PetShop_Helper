# MewHelp 分类器与主题闭环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐语料清洗/增强、完整诊断、分类持久化、主题查询和验收后台。

**Architecture:** 保留第10章既有训练/推理分层，新增小型批处理与仓储模块。分类服务提供版本元数据，分类表和页面共用同一结果来源。

**Tech Stack:** LangChain、SQLAlchemy/MySQL、FastAPI、numpy、ONNX Runtime、HTML/JS；训练依赖仍独立。

**Spec:** [补充设计 §5](../specs/2026-09-28-mewhelp-feature-alignment-design.md)；作业权限复用 [II-1](2026-09-28-mewhelp-alignment-operations-plan.md)，测试约定见[总计划](2026-09-28-mewhelp-alignment-plan.md)。

## Global Constraints

- 17类术语和順序只有 app/core/taxonomy.py 一份权威；黄金预标 exact-set≥80%，各类原始样本至少100。
- 真实文本本地遮蔽后才可处理，外部调用沿用独立授权；错误不兜底成“其他”。
- 划分固定 seed42、80/10/10；增强只增加 train，val/test hash 不变。
- 严档F1≥0.9、中档≥0.8、宽档仅报告；无support的关键类不判通过。
- 8110轻运行时不导入torch；未通过导出一致性不能启动成功；默认批量门槛10，--force才可绕过数量门槛。

## Review Focus

1. 修错把型号/诉求改掉：III-1 标注评估和型号保护。
2. 异步增强没有await或先增强再切分：III-2 预计算适配与holdout hash。
3. 服务两次返回不同版本：III-4/5 拒绝混写，保持原分类。
4. 批次返回少一条却按顺序错位写入：III-5 strict长度/ID校验，零写入。
5. 旧passed报告对应新模型：III-7 显示stale，不判当前PASS。

## 文件边界

语料新增 `scripts/ch10/text_jobs.py`；分类新增 `app/db/topics.py`、`app/topics/batch.py`、`app/topics/client.py`；报告读取新增 `app/topics/acceptance.py`。业务函数不放 API 文件，训练和服务依赖不混装。

### III-1：修错阶段与来源血缘（C01）

**Files:** 新建 `scripts/ch10/text_jobs.py`、`tests/ch10/cleaning_samples.jsonl`、`tests/alignment/test_corpus_cleaning.py`；修改 `scripts/ch10/build_corpus.py`、`scripts/ch10/review_corpus.py`。
**Interfaces:** `async clean_one(text: str, model)->dict` 返回 text/status/reason；`async clean_rows(rows: list[dict], model)->list[dict]` 保留source/review_id及before/after；文件输出 corpus_raw.jsonl/corpus_clean.jsonl/corpus_labeled.jsonl，所有文本均脱敏。

**Test anchor:** `test_cleaning_failure_preserves_review_lineage`: `assert result['text'] == masked_original; assert result['review_id'] == source['review_id']`。

- [ ] 准备标注样例（错字、方言、型号、多诉求、无错）；记录期望保留的诉求/型号。代码RED断言模型错误仍保留原脱敏句、每个来源能追到review_id、无授权时模型调用次数0。
- [ ] 运行 `Test-Alignment tests/alignment/test_corpus_cleaning.py`，确认阶段缺失RED。
- [ ] 在捞池后先按来源ID建立稳定review_id，再接现有脱敏去重后、预标前的修错，修错后再去重；合并重复时保留来源关联；模型改变型号拒绝采用，语义质量用标注集人工核对，不假装正则能证明全部语义保真。
- [ ] GREEN 上述测试与 `tests/ch10/test_corpus.py tests/ch10/test_review_corpus.py`；离线模型只验证数据流，真实Prompt样例报告留IV-2，不用假模型声称达到80%。
- [ ] 留痕、提交并更新C01代码/质量两种状态。

### III-2：训练增强 CLI 接线（C02）

**Files:** 修改 `scripts/ch10/text_jobs.py`、`scripts/ch10/build_dataset.py`；新增 `tests/alignment/test_dataset_augmentation.py`。
**Interfaces:** `async prepare_augmentation(train_rows: list[dict], model, cache_path: Path)->dict[str,dict]` 以原句指纹索引候选/来源/状态；现有同步 `build_dataset(samples, augment_fn=None, seed=42, supplement=None)` 保持同步，CLI先固定划分取train预计算，再以缓存lookup传augment_fn。

**Test anchor:** `test_cli_augmentation_preserves_holdout`: `assert before['val'] == after['val']; assert before['test'] == after['test']; assert set(model_inputs) <= set(train_texts)`。

- [ ] RED：--augment启用时只调用train样本；val/test hash字节不变；失败不加原句；缓存相同模型/Prompt/源hash不二次调用；增强保留parent_review_id，不把coroutine当str。
- [ ] 运行 `Test-Alignment tests/alignment/test_dataset_augmentation.py tests/ch10/test_dataset.py`。
- [ ] 增加 `--augment`、`--augmentation-cache`、`--allow-external-real-text`；缓存绑定模型/Prompt/术语版本；标签从父样本继承，拒绝私人标识/跨集合重复；缺许可/额度明确pending。
- [ ] GREEN 同一命令；固定已审核fixture的CLI前后对比split hash。抽查语义保真样例，发现新增诉求变体不得入训练；真实模型评估留IV-2。
- [ ] 留痕并提交。

### III-3：完整分类质量诊断（C03）

**Files:** 修改 `scripts/ch10/evaluate.py`；新建 `tests/alignment/test_classification_report.py`，复用 `tests/ch10/test_evaluate.py`。
**Interfaces:** 扩展 `evaluate_matrix(gold, predicted)->dict` 保留旧F1字段，新增P/R、每类tn/fp/fn/tp；`build_error_samples(rows, predicted)->list[dict]`；`write_evaluation(report: dict, output: Path)->None` 同源写JSON/Markdown。

**Test anchor:** `test_report_matches_hand_calculated_confusion`: `assert (row['tn'], row['fp'], row['fn'], row['tp']) == (1,1,1,1); assert row['precision'] == row['recall'] == row['f1'] == .5`。字段名统一用precision/recall，接口适配不再用第二套p/r名称。

- [ ] RED：一类tp1/fp1/fn1/tn1得到P=R=F1=.5；多标签错位一条可以贡献多个矩阵错误；无support严格类失败；错误样例不暴露未遮蔽文本。
- [ ] 运行 `Test-Alignment tests/alignment/test_classification_report.py tests/ch10/test_evaluate.py`。
- [ ] 实现新字段、missed/extra/kind、metadata含模型/阈值/术语/test hash；宽档passed=null，阈值红线沿旧规定。调用方不可用“字段存在”冒充实际评估。
- [ ] GREEN 同一命令；对固定矩阵手算和JSON/Markdown做数值核对；不训练模型。
- [ ] 留痕并提交。

### III-4：分类服务元数据与健康检查（C04/C06 基础）

**Files:** 修改 `scripts/ch10/inference_lib.py`、`scripts/ch10/serve.py`；新建 `app/topics/__init__.py`、`app/topics/client.py`、`tests/alignment/test_classifier_metadata.py`；修改 `app/config.py`、`.env.example`。
**Interfaces:** Runtime新增metadata dict；`GET /healthz->{ready,model_version,taxonomy_hash,threshold}`；`POST /classify->{results,model_version,taxonomy_hash,threshold}`保留results；`async classify_batch(texts: list[str], expected: dict)->dict` 验证版本与结果。`CLASSIFIER_BASE_URL`默认http://127.0.0.1:8110，仅本机允许；不自动指向外部文本接收服务。

**Test anchor:** `test_classifier_rejects_version_change_between_batches`: `with pytest.raises(ValueError, match='version'): await classify_batch(texts, expected)`；`test_health_metadata_has_no_local_path`: `assert 'model_dir' not in payload`。

- [ ] RED：缺产物/未通过export不能healthy；标签顺序错拒绝；新旧版本metadata变更能被client识别；返回metadata不含模型绝对路径；服务导入不需要torch。
- [ ] 运行 `Test-Alignment tests/alignment/test_classifier_metadata.py tests/ch10/test_serve.py tests/ch10/test_inference.py`。
- [ ] 在初始化时一次计算模型/分词器/阈值的内容指纹作为model_version，和已有taxonomy_hash一起输出；客户端按现有接口上限每请求最多100条，设置有界超时，检查17类scores/有限数值/标签合法/非空/长度一致；全部分批固定同一版本。
- [ ] GREEN同一命令，fake runtime HTTP演示版本变化被拒；真8110留IV-2。
- [ ] 留痕并提交。

### III-5：分类结果表、迁移与幂等批处理（C04）

**Files:** 修改 `app/db/models.py`、`scripts/ch10/classify_pool.py`；新建 `sql/ch10-topic-classifications.sql`、`scripts/ch10/migrate_topics.py`、`app/db/topics.py`、`app/topics/batch.py`、`tests/alignment/test_topic_batch.py`、`tests/alignment/test_topic_schema.py`。
**Interfaces:** `async list_unclassified(limit: int)->list[dict]`；`async write_classifications(rows: list[dict], meta: dict, run_id: str, reclassify: bool=False)->dict`；`async classify_pool(*,limit=500,min_batch=10,force=False,reclassify=False)->dict`。两表字段完全采用批准spec §5.3；question_id唯一外键，批次run_id主键。

**Test anchor:** `test_batch_is_persisted_once`: `assert first['written'] == 10; assert second['status'] == 'empty'; assert classification_count == 10`。非法结果的“零写入”专指不写分类结果，仍允许写失败批次审计。

- [ ] RED：迁移重复不清数据；10题批次写10，二次empty；9题below_batch不调服务，force可跑；101题分成100+1，顺序与问题ID保持；少/多结果、非法标签、分批版本变化零写入；并发重复不多行；网络失败不留下空标签。
- [ ] 在隔离MySQL运行 `Test-Alignment tests/alignment/test_topic_schema.py tests/alignment/test_topic_batch.py`。
- [ ] 实现脱敏读取、网络调用在写事务外、版本锁定、单批原子写、唯一冲突恢复；--reclassify显式更新当前结果，写含旧/新版本摘要的批次报告；保存原计数JSON兼容读取，不再只计数。
- [ ] GREEN同一命令；执行迁移两遍、fake服务批次两遍核对SQL；对`empty/below_batch/failed/partial/done`分别写报告，不把未完成批次记done。
- [ ] 留痕、提交；注册受review权限保护的classify-pool白名单任务，并追加II-1鉴权测试。

### III-6：主题 API 与页面（C05）

**Files:** 新建 `app/api/topics.py`、`app/static/topics.html`、`app/static/topic-questions.html`、`app/static/topics.js`、`tests/alignment/test_topics_api.py`；修改 `app/db/topics.py`、`app/main.py`。
**Interfaces:** `async topic_distribution()->dict`、`async topic_questions(label: str,offset: int,limit: int)->dict`；GET `/api/topics/distribution` 和 `/api/topics/questions?label=...&offset=0&limit=20`，review token；HTML `/topics`、`/topics/questions`。

**Test anchor:** `test_multilabel_distribution_keeps_unique_question_count`: `assert report['question_count'] == 2; assert sum(report['class_counts'].values()) == 3; assert len(report['class_counts']) == 17`。

- [ ] RED：两题其中一题两标签，独立总数2、标签计数3；17类含零类；分页按classified_at/id稳定无重复；非法标签422；缺token401/未配置503；脱敏列表不含私人号码。
- [ ] 运行 `Test-Alignment tests/alignment/test_topics_api.py`，隔离数据库fixture验证实际聚合。
- [ ] 实现权威表查询、当前分类版本/未分类数、主题图和分页详情；审核关联不存在明确显示，不捏造状态。
- [ ] GREEN同一命令；浏览器点击主题到详情，与SQL对账及验证空池、失败、分页状态。
- [ ] 留痕并提交。

### III-7：九项验收报告后台（C06）

**Files:** 新建 `app/topics/acceptance.py`、`app/api/acceptance.py`、`app/static/acceptance.html`、`app/static/acceptance-data.html`、`app/static/acceptance-eval.html`、`app/static/acceptance-errors.html`、`app/static/acceptance.js`、`tests/alignment/test_acceptance_api.py`；修改 `app/main.py`、`app/core/jobs.py`、`Makefile`。
**Interfaces:** `read_artifact(path: Path, expected: dict)->dict` 返回status/data/reason；`async acceptance_overview()->dict` 返回九项；API `/api/acceptance/overview|data|evaluation|errors`受observability token保护，只读固定根目录，路径不由客户端任意指定。

**Test anchor:** `test_old_passed_artifact_is_stale_for_new_model`: `assert read_artifact(path, new_version)['status'] == 'stale'`；缺文件 `assert result['status'] == 'missing'`。

- [ ] RED：文件不存在missing、坏JSON error、模型/test hash不一致stale、pending如实展示；失败export不能被存在model.onnx覆盖；健康探针失败不影响已有报告；旧无鉴权jobs不可启动这些作业。
- [ ] 运行 `Test-Alignment tests/alignment/test_acceptance_api.py tests/alignment/test_job_access.py`。
- [ ] 实现报告适配，复用III-3/4/5的字段而非重算指标；九项各自有版本/原因/可运行命令，按钮只对应已经实现脚本，真实外发开关不能由任意HTTP参数注入。
- [ ] GREEN同一命令；浏览器确认缺模型和已有真实报告两种状态（fixture不得标现场实测），数据、错例和CLI同源；保存截图。
- [ ] 留痕、提交，写 `docs/mewhelp-alignment-classifier-acceptance.md`；执行 `tests/ch10` 与本批相关测试一次，真实训练仍交IV-2。
