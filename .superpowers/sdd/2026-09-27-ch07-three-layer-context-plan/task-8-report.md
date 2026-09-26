# Task 8 report — offline Chapter 7 acceptance

## RED / GREEN

- Added a near-boundary whole-turn trim test; it passed on the prior implementation and confirms that the last complete turn fits exactly without retaining half of the older one.
- Added a total summary injection test with three immutable segments and a 32-token cap. RED failed because all three segments were injected. GREEN passed after selecting a newest contiguous suffix of whole segments. The same bounded view now feeds budget measurement, `history_ctx`, and model input; `model_ctx` records `omitted_summary_segments`. No old segment is rewritten.
- The new evaluator initially failed to import `app` when invoked as `python scripts/eval_ch07.py`; fixed its local import path. It now writes `data/ch07/reports/offline_eval.json` with **3 pass, 0 fail, 4 pending_upstream**. Its cascade is synthetic, and labeled summary checks validate reference annotations only; neither claims real glm quality.

## Validation and attribution

- Focused Chapter 7 context, summarizer, repository, conversation API, and Graph tests: **46 passed** with isolated MySQL 3307, distinct `mewhelp`/`mewhelp_test` schemas, offline CHAT settings, and explicit legacy `TOKEN_BUDGET=32768` for old Graph fixtures. The migration-twice preservation test is included.
- Default Chapter 7 budget tests, with `TOKEN_BUDGET` removed: **9 passed**. SSE, Ch06 resume, and Graph audit recovery regression: **11 passed**. `git diff --check` exited zero.
- First combined targeted run used the same app/test schema and therefore failed fixture isolation before DB cases; its budget cases were also polluted by the legacy budget override inherited in the shell. This was a test invocation error, corrected by distinct schemas and separate budget run. No product fix followed from that run.
- Prior full-suite output retained in `task-6-full-pytest.txt`: **550 passed, 10 failed, 5 errors**. Milvus at 19530 was down (five setup errors and related retrieval/mining failures); app schema `mewhelp` was absent in affected older paths; the remaining old busy-message, default-config override, demo proxy, and mining assertions are outside the new Chapter 7 changes. A repeat full run would encounter the same unavailable services; focused affected paths above passed.
- Task 7 recorded browser acceptance with an offline API/SSE fixture: two conversations, 52 paged messages, pending-order isolation, stale SSE protection, 503 sidebar fallback, desktop and mobile. This Task 8 run did not repeat that browser session.

## Self-review and limits

- Review focus 1–4 are exercised by the focused repository, summarizer, budget, Graph, SSE, and audit tests. Focus 5 has Task 7's browser evidence. The controller owns independent final review.
- A single newest summary segment that itself exceeds the cap is omitted whole and logged as omitted; the original remains in MySQL. This is the specified memory-coverage limit. `model_ctx` includes the exact injected summary plus omitted count; history and budget use the same cap.
- The original Docker business volume is not mounted. Port 3307 is an isolated test container, not the prior production/demo data. No paid upstream request was sent. Real default/demo twenty-turn dialogues, early-order recall, generated-summary quality, and provider usage comparison remain `pending_upstream` until glm balance is restored.
