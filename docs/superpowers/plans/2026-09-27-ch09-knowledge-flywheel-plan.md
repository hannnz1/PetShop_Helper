# Chapter 9 Knowledge Flywheel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn unresolved customer questions into a deduplicated, human-reviewed knowledge queue, then publish approved answers through the existing MySQL→Milvus lifecycle.

**Architecture:** Keep the existing `low_confidence_questions` table as the raw authority store. Add idempotent feedback capture, an offline model-assisted canonicalization job, an audited review service exposed by a dedicated Bearer API and local CLI, and a retryable publisher that reuses manual knowledge ingestion. Nothing auto-publishes from chat or model output.

**Tech Stack:** Python 3.12, FastAPI, LangChain `with_structured_output`, SQLAlchemy 2.0/MySQL, existing Milvus knowledge pipeline.

**Spec:** `docs/superpowers/specs/2026-09-27-ch09-knowledge-flywheel-design.md`

## Global Constraints

- Use the user's fixed FastAPI/LangChain/MySQL/Milvus stack; Context7 official docs precede any concrete library API implementation, and locked-version signatures are checked locally.
- Real user question text must not be sent to an external model without separate authorization; the offline acceptance uses synthetic labeled samples. No paid glm-5.2 calls while its balance is unavailable.
- Existing retrieval confidence and `useful=false` writes stay unchanged. No new arbitrary 0.5 score threshold.
- `KNOWLEDGE_REVIEW_TOKEN` is separate from observability and existing unprotected admin routes. Missing token refuses administration.
- Only a human-entered, approved answer reaches knowledge ingestion. MySQL is authoritative; vectorization is independently retryable.
- Every completed stage appends four fields to `dev-notes/ch09.md`: user wording, key output, corrections/rejections, failures/rework. TDD for code; labeled samples for Prompt quality.

## Review Focus

1. Duplicate feedback for the same assistant message returns one raw record, while feedback on another user's conversation is refused.
2. A model returning a `matched_question_id` outside its supplied candidates does not merge unrelated questions.
3. Two workers claiming the same raw row never double-count one occurrence.
4. A reviewer retry, double click, or failure between knowledge insertion and queue-link update does not create duplicate knowledge.
5. A configured external chat endpoint with no explicit real-text authorization makes the canonicalization job report `pending_upstream` without sending any raw question.

## File map

| Unit | Responsibility |
| --- | --- |
| `sql/ch09-flywheel.sql`, `app/db/models.py`, `app/db/flywheel.py` | Additive schema, unique links, transactional queue/review/publish state |
| `app/flywheel/privacy.py`, `app/flywheel/canonicalize.py` | Local masking, bounded candidates, Pydantic extraction and validation |
| `app/flywheel/review.py` | Review state machine, audit and knowledge publishing orchestration |
| `app/api/flywheel.py`, `app/main.py`, `app/config.py` | Feedback route and protected review routes, isolated admin token |
| `scripts/ch09/flywheel_canonicalize.py`, `scripts/ch09/flywheel_review.py` | One-shot batch and operator CLI |
| `tests/flywheel/`, `docs/ch09-flywheel-acceptance.md`, `dev-notes/ch09.md` | RED/GREEN, labeled Prompt verification, reproducible acceptance |

### Task 1: Raw feedback and additive queue schema

**Files:** Modify `app/db/models.py`, `app/db/repository.py`, `app/config.py`, `app/main.py`, `tests/conftest.py`; create `app/db/flywheel.py`, `app/api/flywheel.py`, `sql/ch09-flywheel.sql`, `tests/flywheel/test_feedback.py`, `tests/flywheel/test_schema.py`.

**Interfaces:** `record_unresolved_feedback(user_id: str, conversation_id: int, assistant_message_id: int) -> int`; `POST /api/feedback/unresolved` with these fields. Produces `CanonicalQuestion`, `CanonicalOccurrence`, and `FlywheelReviewAction` ORM tables and an additive migration. Existing auto-written raw rows remain compatible. Define a unique feedback source reference and unique `raw_question_id` occurrence link.

- [ ] **Step 1:** Context7-check FastAPI request validation and SQLAlchemy 2.0 async MySQL unique constraints/transactions. Add tests for owned assistant message, wrong owner/role, nearest user question, duplicate POST, and old raw rows.
- [ ] **Step 2:** Run focused tests RED. Implement additive DDL/ORM and repository operation, then API with the same demo `user_id` trust boundary as current chat. Run focused tests GREEN.
- [ ] **Step 3:** Apply migration only to isolated MySQL, verify it can rerun without touching current knowledge rows; run affected conversation/Graph tests. Append task notes and commit.

### Task 2: Canonicalization, candidate checking, and idempotent linkage

**Files:** Create `app/flywheel/__init__.py`, `app/flywheel/privacy.py`, `app/flywheel/canonicalize.py`, `scripts/ch09/flywheel_canonicalize.py`, `tests/flywheel/test_canonicalize.py`, `tests/flywheel/canonicalization-labels.json`; modify `app/db/flywheel.py`, `app/config.py`.

**Interfaces:** `canonicalize_batch(limit: int, model: BaseChatModel, *, allow_external_real_text: bool = False) -> BatchResult`; Pydantic result has `canonical_question`, `draft_answer`, `matched_question_id: int | None`, `reason`. One `CanonicalOccurrence.raw_question_id` is unique; occurrence count is derived from links. Limit candidate IDs and validate model choice against exactly the sent set. Local exact normalized duplicate may merge deterministically.

- [ ] **Step 1:** Context7-check locked LangChain `with_structured_output` and PromptTemplate interfaces. Build labeled pairs (same question/different phrasing, near but different policy, invalid ID, PII) and run the Prompt/result validator on the set before enabling batch writes.
- [ ] **Step 2:** Add RED code tests for bounded candidates, local masking, unsupported structured output, external-endpoint guard, invalid match ID, concurrent/repeated raw ID, and occurrence count. Implement minimal batch + repository linkage; run GREEN without network.
- [ ] **Step 3:** CLI emits processed/merged/new/pending counts, no raw text or secrets. Verify same synthetic batch twice is idempotent. Append task notes and commit.

### Task 3: Protected human review API and CLI

**Files:** Modify `app/api/flywheel.py`, `app/config.py`; create `app/flywheel/review.py`, `scripts/ch09/flywheel_review.py`, `tests/flywheel/test_review.py`.

**Interfaces:** `GET /api/review/questions` paged queue; `POST /api/review/questions/{id}/decision` with `action: reject|defer|merge|approve`, reason/category/approved answer as required. `KNOWLEDGE_REVIEW_TOKEN: SecretStr | None` protects both routes with 503 when unset and 401 when missing/wrong. `review_question(..., request_id: str) -> ReviewResult` records append-only action and enforces state transitions/idempotency.

- [ ] **Step 1:** Context7-check FastAPI HTTPBearer dependency behavior and SQLAlchemy transaction/row locking. Tests RED for token cases, paging without secrets, invalid transitions, reason requirements, human answer requirement, repeated/conflicting request IDs, and merge target validity.
- [ ] **Step 2:** Implement service/API and local CLI invoking the same service; run focused tests GREEN. CLI outputs IDs/statuses, masks raw PII, and never prints Bearer values.
- [ ] **Step 3:** Run affected admin/API tests, append task notes, commit.

### Task 4: Approved knowledge publication and retry

**Files:** Modify `app/flywheel/review.py`, `app/db/flywheel.py`, `app/api/flywheel.py`, `scripts/ch09/flywheel_review.py`; create `tests/flywheel/test_publish.py`.

**Interfaces:** `publish_approved(canonical_id: int, request_id: str) -> PublishResult`. Reuse `app.kb.dualwrite.write_manual_report` and its normalized question/answer identity; save linked `knowledge_chunk_id`. Return `approved_pending_vector` until existing `vectorize_pending` marks the chunk done. Retry recovers the same chunk after a crash between insert and link update.

- [ ] **Step 1:** Context7-check any newly used SQLAlchemy/Milvus API and read the current manual-ingest contract. Tests RED for unapproved publish rejection, twice-publish idempotence, failure after chunk insert, vectorization failure then retry, and no model draft used as answer.
- [ ] **Step 2:** Implement publish saga and recovery with existing MySQL knowledge lock; run focused tests GREEN. Do not add an automatic chat-path publisher.
- [ ] **Step 3:** Run isolated MySQL + fake vectorization integration, append task notes, commit.

### Task 5: Integrated offline acceptance and independent review

**Files:** Create `docs/ch09-flywheel-acceptance.md`; modify `dev-notes/ch09.md` and targeted tests as findings require.

**Interfaces:** Reproducible command sequence for three intake sources, two variants to one canonical ID, rejected/merged/deferred/approved cases, MySQL pending chunk and vectorized search when services are available, plus explicit `pending_upstream` for paid model/service gaps.

- [ ] **Step 1:** Run labeled Prompt set and all `tests/flywheel` plus affected Graph/KB/API tests in isolated MySQL, with no paid model calls. Run CLI and protected API path with synthetic data; check secrets/PII absent from logs/trace.
- [ ] **Step 2:** Run real local Milvus/embedding only if configured and previously authorized data scope allows; otherwise document exact missing prerequisite. Preserve existing DB/collections.
- [ ] **Step 3:** Use `superpowers:requesting-code-review` for whole-branch review; fix Important/Critical findings with `superpowers:receiving-code-review`, rerun affected tests, append review/finish four-field notes and commit acceptance report.

