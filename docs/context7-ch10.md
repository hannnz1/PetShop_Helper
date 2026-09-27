# 第 10 章接口核对（2026-09-28）

- Task 1 只使用 Python 标准库，本地术语表源于用户指定的课程仓库；无第三方 API 变更。
- Task 2 经 Context7 查询官方 [LangChain Python ChatOpenAI 文档](https://docs.langchain.com/oss/python/integrations/chat/openai)：`with_structured_output(PydanticModel, method=...)` 返回结构化对象；Prompt 用 `ChatPromptTemplate.from_messages`，异步执行用 `ainvoke`。本机锁定 `langchain-core 1.6.4`、`pydantic 2.13.5`；用假模型验证 Pydantic 结果与失败路径。
- Task 2 经 Context7 查询官方 [SQLAlchemy 2.0 文档](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)：`AsyncSession` 对 `select(...).limit(...)` 异步读取；本机锁定 `sqlalchemy 2.0.54`。低置信池只读并在本机脱敏。

后续 Transformers、PyTorch、ONNX Runtime、tokenizers、FastAPI 的具体实现必须按各任务再查 Context7 与锁定版本，不沿用课程旧代码的接口假设。

- Task 4 经 Context7 查询官方 [Transformers Trainer 文档](https://huggingface.co/docs/transformers/main_classes/trainer)：当前接口为 `eval_strategy`、`save_strategy`、`Trainer` 的 `compute_metrics` 与回调；[PyTorch BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) 要求多标签浮点目标；[scikit-learn F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) 支持 multilabel micro 聚合。训练依赖放可选 `ml` 组，当前环境未安装，不因 30 条合成样本下载模型。
- Task 5 经 Context7 查询官方 [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)：`torch.onnx.export` 支持 `opset_version`、`dynamic_axes`、`dynamo=False`；[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)：`InferenceSession(..., providers=["CPUExecutionProvider"])` 与 `session.run(None, inputs)`。本环境无模型/重依赖，导出只完成离线逻辑验证。
- Task 6 经 Context7 查询官方 [FastAPI 请求体与生命周期](https://fastapi.tiangolo.com/tutorial/body/)：Pydantic 请求模型、`FastAPI(lifespan=...)` 和 `TestClient` 上下文；[Tokenizers 批量编码](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html)：`encode_batch`、padding 与 attention mask。ONNX Runtime 的 `session.run(None, inputs)` 由 Task 5 官方文档再次沿用，推理依赖惰性加载。
