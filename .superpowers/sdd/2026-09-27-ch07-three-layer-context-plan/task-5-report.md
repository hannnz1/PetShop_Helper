# Task 5 report — async append-only segmented summaries

## RED evidence

- `python -c "from app.core.summarizer import summarize_pending"` failed with `ModuleNotFoundError` before implementation.
- Initial `pytest -q tests/test_ch07_summarizer.py` crashed during Windows native extension imports before collection. Retrying with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and `-p pytest_asyncio.plugin` produced actionable failures.
- `test_nonstream_invoke_schedules_only_new_completed_audit` failed because a pending result reused an audit marker and scheduled twice (`[7, 7]` versus `[7]`).
- `test_restart_candidate_scan_finds_unprocessed_layer2` failed because `list_summary_candidates` was absent.
- A direct assertion on `_facts('无明确事实。')` failed, exposing punctuation handling for a no-facts marker.

## GREEN evidence

- Isolated environment: `CHAT_MODEL=offline-test`, `CHAT_BASE_URL=http://127.0.0.1:9/v1`, `CHAT_API_KEY=offline-test`, MySQL app and `*_test` schemas on `127.0.0.1:3307`; no paid upstream calls.
- `pytest -p pytest_asyncio.plugin -q tests/test_ch07_summarizer.py tests/test_ch07_repository.py tests/graph/test_ch07_context.py tests/graph/test_ch06_resume.py` with optional plugin autoload disabled: **33 passed, 1 Starlette deprecation warning**.
- Four JSONL labeled references were checked individually for required and prohibited terms: 4/4 labels internally consistent. This validates the evaluation data, not LLM summary quality. Real Prompt quality and 20-turn GLM acceptance remain `pending_upstream`.
- `git diff --check`: exit 0 (Git reported only LF-to-CRLF conversion notices).
- Full offline suite with the same test environment: **495 passed, 58 failed, 5 errors, 1 warning in 454.65 s**. First isolated chat API failure is `ContextBudgetExceeded required=17627, window=2000` at startup, documented after Task 4. Milvus `127.0.0.1:19530` connection refusal caused setup errors. Other failures span legacy app startup/configuration, live CLI proxy, and mining/live integrations; no Task 5 focused test failed. Full suite did not pass.

## Implementation and self-review

- Token threshold uses the calibrated estimator on Layer2 user text plus shortened assistant text. A batch contains only new completed visible turns through the Layer2 anchor. Previous summary content never enters its Prompt.
- `commit_summary_segment` locks the conversation row, compares the old anchor, checks the completed assistant endpoint, inserts one new sequence, and advances the anchor/projection in a single transaction. Concurrency tests observed exactly one committed range. MySQL triggers reject direct anchor rewind and summary update/delete.
- Empty fact output writes an empty immutable segment and advances the anchor; `context_layers` already excludes empty segments from model input. Model errors leave the anchor unchanged for a later turn or restart scan.
- SSE schedules only after the `[DONE]` frame resumes and the graph has exhausted normally. Disconnects, pending interrupts, and graph/audit errors do not reach that call. Non-stream invoke requires a newly advanced audit marker. Summary task work never changes the already produced reply.
- Startup scans persisted Layer2 gaps; shutdown waits at most five seconds, cancels lingering tasks, then waits one additional second. `trigger/start/done/skip/fail` events use the existing local `log/app.log` context logger.
- Remaining limit: the configured `LAYER2_SUMMARY_TOKENS` threshold is a dedicated Layer2 ceiling (default 1695, demo target), while per-route model budgets may be smaller; the model view still applies its own final trim. Prompt factual quality cannot be established without the unavailable upstream model.
