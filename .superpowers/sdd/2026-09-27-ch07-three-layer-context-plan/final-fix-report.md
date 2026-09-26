
## Final review rework started

The prior offline_cascade_transitions case calls repository.advance_layer1 explicitly and does not prove automatic Graph entrypoint downgrade. Final fix wave replaces that evidence with real zero-anchor Graph invoke/stream/resume regressions. Fake summary output remains offline evidence only. Full prior result: 555 passed, 10 failed, 5 errors. New RED/GREEN and complete gate output will be recorded here.

## Fixes and evidence (updated during verification)

- Important 1: GraphRuntime now prepares the authoritative snapshot in invoke, stream, and resume entrypoints after ownership/audit/pending validation. It computes a conservative route allocation from injected settings, moves only the oldest complete persisted visible pairs via advance_layer1, then reloads the snapshot before the graph sees it. Current interrupted/tool exchanges remain checkpoint-only. Completed-turn scheduling remains at its existing successful invoke/SSE completion boundary.
- Important 2: summary content accepts strings, lists of strings, and text blocks with string text. Unknown/malformed blocks and empty lists fail inside the worker error boundary, leaving the summary anchor and append-only table untouched and retryable.
- Minor 1: has_summary queries nonempty persisted segments rather than latest projection; API fixture now writes a real summary segment.
- Minor 2: model_ctx records ToolMessage.tool_call_id.
- Minor 3: context and usage records share a unique call_id, phase, conversation and step; usage records include estimated input including tool schemas, provider input/output/total, and input delta. Stream chunks accumulate standard usage counters and final-answer usage contributes to turn totals. Missing provider usage is represented by empty usage/null delta; no real calibration claimed.

RED: final-fix-red.txt captures six expected failures: invoke/stream kept anchor zero; supported text blocks were discarded; malformed/image/empty-list content silently skipped. Interim run captures badge false, missing tool ID, and missing final usage as three expected Minor failures; valid resume was subsequently isolated from the separate repeated-invalid-resume diagnostic.
GREEN: first core/context gate 38 passed. Strengthened actual business-model cascade plus all summary/context/API/pending-resume tests: 46 passed (final-fix-focused.txt). Cascade starts from zero anchors, uses real Graph/checkpoint/MySQL and asynchronous worker with deterministic fake models, verifies three append-only segments across four turns, refreshed history snapshots, injected summaries in actual model calls, unchanged first segment, disjoint raw batches, and audit/checkpoint retention. No patched scheduler or manual advance_layer1 occurs in this cascade regression.

Self-review: complete-pair selection uses the existing whole-turn trimming rules; boundary writes stay transactional/monotonic; context is reloaded even after a lost boundary race; unsupported output cannot reach commit_summary_segment. The early allocation is deliberately conservative before intent is known; actual model call budgets still recalculate and enforce the final cap. Historical raw messages are never edited. No paid upstream call, migration or original app database write is part of this wave.

Docs: attempted Context7 resolve; unavailable to model. Official web fallback transport also failed. Reused existing library fields/APIs only and recorded the limitation in docs/context7-ch07.md.

## Final gates and commands

All Python commands use `..\..\.venv\Scripts\python.exe` from this worktree, with escalated execution because the existing virtualenv needs it. CHAT_MODEL=offline-test, CHAT_BASE_URL=http://127.0.0.1:9/v1, CHAT_API_KEY=offline-test; DATABASE_URL points at isolated MySQL 127.0.0.1:3307/mewhelp and TEST_DATABASE_URL at distinct mewhelp_test; PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; PYTHONIOENCODING=utf-8 on final runs. Only disposable test schema fixtures perform integration writes.

- RED: `-m pytest -q -p pytest_asyncio.plugin tests/test_ch07_summarizer.py -k 'content_blocks or zero_anchor' --tb=short`: 6 failed as intended, saved final-fix-red.txt.
- Initial GREEN: summarizer/context files: 38 passed (final-fix-green.txt). Stronger business-route/actual-model-view gate adding API and existing resume files: 46 passed (final-fix-focused.txt).
- Full suite: `-m pytest -q -p pytest_asyncio.plugin --tb=short`, TOKEN_BUDGET=32768 for parity with prior full gate: **564 passed, 10 failed, 5 errors, 1 warning**, 207.38s (final-fix-full-pytest.txt). Failure/error identifiers exactly match task-8-full-pytest.txt, verified by Compare-Object after stripping exception suffixes. This is NOT a clean full suite. The additional list-of-strings summary parameter and unique call_id correlation assertions were verified in the subsequent final focused gate; the full run had already imported its modules before those last instrumentation/test refinements.
- `scripts/eval_ch07.py --offline`: **4 pass / 0 fail / 4 pending_upstream** (final-fix-eval.txt and tracked data/ch07/reports/offline_eval.json). The real automatic invoke/stream cascade and valid pending resume are now in its five-test subprocess, alongside failure-retry and concurrent commit checks.
- A mixed final focused invocation without explicitly removing inherited TOKEN_BUDGET produced 67 passed / 4 default-budget failures (final-fix-final-focused.txt). Corrected by separate process with `Remove-Item Env:TOKEN_BUDGET`: **9 default-budget tests passed** (final-fix-default-budget.txt). This was test environment contamination, not a production failure; no budget formula change was made.
- Final remaining Chapter 7 gate uses explicit TOKEN_BUDGET=32768 and the files summarizer, graph context, conversations API, graph Ch06 resume, layers, repository, evaluator. See final-fix-final-green.txt for final count.
- `git diff --check` passed (only normal Windows CRLF notices).

Full-suite failure attribution (same as prior gate): unavailable Milvus 127.0.0.1:19530 and missing original app schema affect live vector/retrieval/mining paths. Legacy busy-message, explicit-budget default-config, and demo proxy assertions remain. Existing Starlette BlockingPortal deprecation warning remains. Exact failure/error identifiers follow:
FAILED tests/test_chat_api.py::test_stream_failure_after_tool_status_has_no_done[failure1-\u4f1a\u8bdd\u6b63\u5728\u5904\u7406]
FAILED tests/test_config.py::test_defaults_and_secret_type - AssertionError: ...
FAILED tests/test_demo_eval.py::test_live_cli_clients_ignore_proxy_environment[scripts.demo_chat]
FAILED tests/test_milvus_hybrid.py::test_existing_incompatible_hybrid_collection_is_rejected
FAILED tests/test_mining.py::test_offline_source_attribution_and_skip_existing
FAILED tests/test_mining.py::test_offline_empty_source_is_not_reextracted_on_rerun
FAILED tests/test_mining.py::test_offline_grouped_extraction_keeps_real_source_refs
FAILED tests/test_mining.py::test_offline_private_identifier_echo_is_not_staged_as_knowledge
FAILED tests/test_mining.py::test_offline_user_cannot_spoof_assistant_quote
FAILED tests/test_retrieval_ch04_live.py::test_four_strategies_over_real_isolated_collection
ERROR tests/test_dualwrite_ch04.py::test_vectorize_writes_bm25_and_metadata
ERROR tests/test_dualwrite_ch04.py::test_flush_failure_keeps_mysql_pending_for_retry
ERROR tests/test_milvus_hybrid.py::test_bm25_hits_model_number - pymilvus.exc...
ERROR tests/test_milvus_hybrid.py::test_dense_and_hybrid_return_scored_hits
ERROR tests/test_milvus_hybrid.py::test_category_filter_is_applied_to_both_routes

## Separate baseline limitation: repeated invalid resume

Exploratory negative test: create refund conversation, pause at order selection, resume with `invalid`, receive another select_order interrupt, then resume with owned order 1001. Both base 146f7b7 GraphRuntime and updated GraphRuntime reject the second resume with GraphDivergence on a no-history conversation (final-fix-resume-diagnostic.txt: two expected diagnostic failures). The earlier prior-history variant rejected with ResumeNotPending. This is a pre-existing repeat-interrupt/checkpoint issue, not caused by layer downgrade; retained without expanding this final fix wave per controller instruction. A normal first valid resume, pending new-turn rejection, existing wrong-value interrupt behavior, and safe summary timing all pass their focused regressions.

Reproducer retained as diagnostic-invalid-resume.py and diagnostic-baseline-runtime.py in this report directory. To rerun, copy the former temporarily to tests/_diagnostic_resume_final.py and the latter to .baseline_runtime.py at worktree root, then run pytest on that test under the same offline isolated environment. Remove the temporary copies afterward. Baseline source was captured with `git show 146f7b7:app/graph/runtime.py` before execution; no checkout replacement was used.

## Remaining concerns / re-review handoff

- Real glm default/demo twenty-turn acceptance, early-order recall, generated-summary factual quality, and measured provider calibration remain pending_upstream. Fake models do not establish real summary quality.
- Default allocation preserves the existing budget policy; the pre-routing downgrade uses the conservative minimum of supported routes. Actual model input continues to recompute route/evidence/summary costs per call.
- No UI change or new browser acceptance run in this wave. Earlier Task 7/8 evidence is not relabeled as newly run.
- Context7 and official web fallback were unavailable; no new library API was introduced, and the failure is recorded rather than claimed as successful lookup.
- Controller should perform the one scoped final re-review against the fix commit. No subagent/reviewer was spawned in this worker.

Final affected gate completed: **62 passed, 1 existing deprecation warning** in 19.18s (final-fix-final-green.txt). Together with the separately clean default-budget gate, **71 affected tests passed**. No remaining final-review finding is intentionally deferred; unrelated baseline invalid-repeat-resume and full-suite/environment limitations are stated above.

Implementation commit: afdb83844707bf1e357f6edd6eac4b1bc6800896 (base 146f7b7). Final report/evidence committed separately after verification.
