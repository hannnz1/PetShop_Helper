# SDD ledger — plan: docs/superpowers/plans/2026-09-27-ch07-three-layer-context-plan.md

Spec: docs/superpowers/specs/2026-09-27-ch07-three-layer-context-design.md (approved 2026-09-27)
Execution: user chose per-task subagent implementation and review.
Workspace: isolated .worktrees/ch07-context, branch feat/ch07-context, base 9c36b00.
Baseline: 20 passed in 5.83s (memory, audit recovery, Ch06 resume) with offline CHAT_* placeholders and isolated MySQL port 3307. First collection failed because worktree has no ignored .env; no real model calls were made.
Tooling: native create_worktree returned “Not a git repository” because chat cwd is the parent directory; manual git worktree used. Git Bash sdd-workspace script cannot create Windows C:/ path; stopped it and created this plan-scoped ignored workspace via PowerShell. No project files were affected by that script failure.

## Preflight task/interface scan

| Tasks | Shared file/interface and finding |
| --- | --- |
| 1→3 | `ContextSnapshot` in repository feeds layer builder; defined with visible rows after summary anchor. Compatible. |
| 1→5 | repository anchors/segments feed summarizer. Lock+append transaction belongs Task 5; Task 1 only schema/read/Layer1 advance. Compatible. |
| 1→6 | repository session ownership and visible rows feed paged GET. Task 1 snapshot is internal, Task 6 gets separate bounded read functions. Compatible. |
| 2→3 | `ContextBudget` feeds builder; single CJK estimator must be shared. Compatible. |
| 2→4 | budget self-check and runtime errors; graph must preserve legacy behavior until Task 3 integrated. Compatible. |
| 3→4 | model/history context functions feed nodes; current in-progress Graph messages not yet in MySQL and must be passed separately. Compatible. |
| 4→5 | runtime/SSE integration shared; Task 5 adds post-completion summary scheduling only after Task 4 keeps completion semantics. Compatible. |
| 4→8 | Graph behavior and logs feed eval. Compatible. |
| 5→8 | summary outcomes feed eval. Compatible. |
| 6→7 | GET API contract feeds UI, old tools filtered. Compatible. |
| 7→8 | browser acceptance follows UI. Compatible. |
| 1 | Test expects new Graph audit rows without tools; implementation changes `append_turn_messages`, while legacy agent `append_message` remains. Compatible. |
| 2 | Budget test expects measured, not hard-coded 5650; exact numeric bound may need calibration. Compatible. |
| 3 | Current tool exchange comes from checkpoint even though MySQL visible snapshot excludes tools. Compatible. |
| 4 | `coref` is START node on all paths, so each route can log `history_ctx`. Compatible. |
| 5 | Plan names SSE scheduling but spec says completed user reply; non-stream invoke path also needs scheduling. Ruling: include both completed graph entrypoints because the spec governs; if wrong, extra offline summary attempts only on a successful invoke turn. |
| 6 | GET ownership matches existing local demo user_id, not real authentication. Compatible with spec. |
| 7 | UI exception applies to page only; back-end Task 6 tests remain mandatory. Compatible. |
| 8 | Real glm balance unavailable; mark `pending_upstream`, no paid request. Compatible. |

Task 1: review ⚠️ summary-anchor monotonicity and append-only write behavior belongs to Task 5; carry forward there.
Task 1: minor (deferred): focused API regression emitted one pre-existing Starlette TestClient deprecation warning; final review should triage.
Task 1: complete (commits 9c36b00..7f09d8e, review clean). Focused isolated MySQL regression 13 passed; full suite had environment-dependent failures recorded in task-1-report.md.
Task 2: initial review spec ❌ / quality Needs fixes. Important: `current_peak` reserves user input and tool results but omits accumulated assistant/tool-call messages. Fix round 1 pending. ⚠️ Startup validation with rendered prompt/tool schema belongs to Task 4.
Task 2: fix round 1/5 (0 addressed, 1 open — separate `final_answer` model invocation still sees MAX_AGENT_STEPS assistant outputs, while formula reserves only MAX_AGENT_STEPS-1; commits 4848331..0edd2a4).
Task 2: fix round 2/5 (1 addressed, 0 open — all assistant outputs and preceding tool results now reserved; commits 0edd2a4..6d4240c).
Task 2: complete (commits 7f09d8e..6d4240c, review clean). 49 focused budget/memory/config tests passed. Task 4 must wire startup validation with actual prompt/tool costs and enforce per-step aggregate tool-result cap.
Task 3: Ruling: allow Task 3 to modify app/config.py and add `layer2_assistant_chars` (default 48, positive) — approved spec says Layer2 assistant prefix length is configurable, but plan file map omitted the setting — if wrong, the only cost is a reversible extra config field/default.
Task 3: minor (deferred): new trim test uses zero-token budget only; add a near-boundary whole-turn case if Task 8 regression leaves doubt. Final review should triage.
Task 3: review ⚠️ whole-suite result has 13 fails and 5 setup errors; focused 56 passed. Task 8 must attribute environmental failures and resolve in-scope regressions.
Task 3: complete (commits 6d4240c..560aca0, review clean). Context assembly ready for Task 4.
Task 4: initial review spec ❌ / quality Needs fixes. Important: graph routing uses global get_settings().max_agent_steps while budget uses injected runtime settings; fix round 1 pending.
Task 4: minor (deferred): model_ctx omits ToolMessage.tool_call_id, making tool reply correlation less precise; final review should triage.
Task 4: minor (deferred): focused run emits an existing Starlette deprecation warning; final review should triage.
Task 4: review ⚠️ background summary belongs Task 5, real model acceptance remains pending_upstream.
Task 4: Ruling: isolate legacy /api/extract endpoint tests from FastAPI lifespan when they intentionally set TOKEN_BUDGET below one Graph turn, and separately assert startup rejection — approved Ch07 spec requires startup self-check — if wrong, old endpoint integration coverage under that invalid configuration is reduced.
Task 4: fix round 1/5 (1 addressed, 0 open — graph routing now uses injected max_agent_steps; commits 041986a..6f9108a).
Task 4: complete (commits 560aca0..6f9108a, review clean). Focused Graph/SSE/Ch06 regression 64 passed after fix; original combined targeted run 86 passed. Full-suite environmental failures remain for Task 8 attribution.
Task 5: initial review spec ❌ / quality Needs fixes. Important: trigger uses fixed/global Layer2 threshold instead of actual derived/injected allocation; worker read/settings failures occur outside handler; unbounded full-original backlog can make retries fail forever. Fix round 1 pending.
Task 5: minor (deferred): unsupported non-string summary output can advance empty anchor; SSE lifecycle test uses patched scheduler only; scratch report accidentally tracked; Starlette warning. Final review to triage.
Task 5: review ⚠️ full 58 failures/5 errors lack retained individual output; Task 8 must rerun/attribute. Prompt labels only validate data, true model quality pending_upstream. Total summary injection cap remains to verify downstream.
Task 5: fix round 1/5 (3 addressed, 0 open — derived/injected Layer2 threshold, full worker failure boundary, bounded oldest-first raw batch; commits a457ff9..51688d3).
Task 5: complete (commits 6f9108a..51688d3, review clean). Focused 37 + 19 offline tests passed; full-suite failures remain for Task 8 attribution.
Task 6: initial review spec ❌ / quality Needs fixes. Important: conversations.updated_at does not advance on normal message writes, so newest-first list can be wrong. Fix round 1 pending.
Task 6: minor (deferred): consider `(user_id, updated_at, id)` index for high-volume list sorting; final review triages.
Task 6: fix round 1/5 (1 addressed, 0 open — completed turn now updates parent timestamp transactionally; commits ccbb20c..805fff3).
Task 6: complete (commits 51688d3..805fff3, review clean). Focused API/repository/audit tests 12 passed; full-suite output saved in task-6-full-pytest.txt for Task 8 attribution.
Task 7: complete (commits 805fff3..389d395, Vibe Coding page exception). Browser fixture verified two conversations, 52 paged messages, pending-order isolation, stale SSE protection, sidebar 503 fallback, 1280px desktop and 390px mobile; node --check and diff --check passed. No page TDD/code review per user's explicit exception; back-end contract was reviewed in Task 6.
Task 8: initial review spec ❌ / quality Needs fixes. Important: default_20_turns only checks token estimate, not actual builder retention; demo_22_turn_cascade uses handcrafted anchors/summary and overclaims transition and recall. Fix round 1 pending.
Task 8: review ⚠️ applicable full suite and browser operations were cited from earlier tasks, not repeated after Task 8 changes; controller requires final evidence. Minor: evaluator inherits potentially contaminating env settings.
Task 8: fix round 1/5 (2 addressed, 0 open — actual 20-turn model view asserted, synthetic post-cascade rendering named honestly plus separate offline transition tests; commits e73c86c..146f7b7).
Task 8: complete (commits 389d395..146f7b7, review clean). Evaluator 4 pass/0 fail/4 pending_upstream; affected 20 tests passed. Captured full offline suite 555 passed/10 failed/5 errors due documented service/legacy conditions; final reviewer must assess residuals. Task 7 retained multi-conversation browser evidence; Task 8 separately repeated Chrome refund/SSE gate.
Final whole-branch review at 146f7b7: Not ready to merge. Important: no production caller for advance_layer1, so automatic cascade never occurs; unsupported non-string summary content advances an empty immutable segment. Minor: has_summary projection can hide prior factual segments; model_ctx omits ToolMessage.tool_call_id; provider usage not correlated with per-call estimates (real calibration pending). One final fix wave pending, then one scoped re-review.

Final fix wave: both Important findings and all three Minor findings implemented. Core RED six failures captured; stronger Graph invoke/stream model cascade plus summary/context/API/resume gate 46 passed. Full-suite run and final evaluator evidence pending in final-fix-report.md; controller owns scoped re-review/refork.

Final rework result: 62 final affected tests passed plus 9 default-budget tests passed in separate clean process; evaluator 4 pass/0 fail/4 pending_upstream. Full offline gate 564 pass/10 fail/5 error has the identical pre-existing failure/error identifiers as Task 8. Preserved invalid→valid resume reproducer shows the same rejection on base 146f7b7 and current runtime; kept as separate baseline concern. All five final-review findings addressed; ready for controller's one scoped re-review.
