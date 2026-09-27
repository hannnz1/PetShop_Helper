# Chapter 10 Topic Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 17-class multilabel classifier pipeline from the Chapter 9 question pool to a separately served ONNX model, while making missing data, quota, or compute visible.

**Architecture:** One taxonomy module owns names, boundaries and label IDs. Offline corpus/dataset jobs isolate private text and preserve human review and split lineage; an optional ML dependency group trains and exports a model. A light FastAPI process serves ONNX results on port 8110 and writes reproducible evaluation reports. The current chat Graph does not call this classifier.

**Tech Stack:** Python 3.12, SQLAlchemy/MySQL, LangChain/Pydantic, Transformers/PyTorch, scikit-learn, ONNX Runtime/tokenizers/numpy, FastAPI.

**Spec:** `docs/superpowers/specs/2026-09-27-ch10-topic-classifier-design.md`

## Global Constraints

- Keep `app/core/taxonomy.py` the only 17-class authority; copy exact names/order/boundaries/examples/severity from `C:\Users\Administrator\Desktop\project\MewHelp\app\core\taxonomy.py`, preserving source attribution as appropriate. Every label table derives from it.
- Never externalize real `low_confidence_questions` text without separate authorization. Default external model jobs to `pending_upstream`; run offline acceptance on synthetic labeled samples.
- A model failure is not an “其他” label. Reject invalid labels and record the failed item. Batch prelabel requires a golden exact-set accuracy gate of 80% on reviewed examples.
- Split original samples by label combination before augmentation; only training grows. Fixed seed 42 and dataset/taxonomy hashes are part of outputs.
- `hfl/chinese-roberta-wwm-ext`, 17 float labels, `problem_type="multi_label_classification"`, validation micro-F1 threshold sweep 0.30–0.70 by 0.05, strict-better tie rule, and ONNX opset 17 are fixed by the approved spec.
- Keep heavy ML dependencies out of the chat runtime; ONNX serving at `127.0.0.1:8110` reads the same taxonomy/threshold logic as evaluation. Old course reports are references only.
- Use Context7 official docs and check locked local versions before each library-specific implementation; code tasks use RED→GREEN, pure Prompt/data quality uses labeled evaluation. Append four-field notes to `dev-notes/ch10.md` at each completed stage.

## Review Focus

1. A phone/order/social ID in a raw pool question cannot appear in model input, logs, reports or audit export; test masking and a failed model call.
2. A failed or invalid model prelabel must not silently become “其他” and inflate that class; test failure records and batch gate behavior.
3. A paraphrase identical to a validation/test original cannot leak into training; test hashes across all three splits.
4. A label order mismatch between a trained artifact and the deployed taxonomy must reject service startup; test metadata validation.
5. An ONNX export with one sample whose thresholded label set differs from fresh Torch inference must write a failed report and refuse success; test mismatch path.

## File map

| Unit | Responsibility |
| --- | --- |
| `app/core/taxonomy.py`, `scripts/ch10/corpus_lib.py` | Single 17-class authority and local masking/dedupe/stratification |
| `scripts/ch10/build_corpus.py`, `prelabel.py`, `validate_golden.py`, `golden_samples.jsonl` | Pool intake, opt-in model jobs, verified labels and human audit export |
| `scripts/ch10/build_dataset.py`, `supplement_sizefit.jsonl` | Fixed split, training-only augmentation/supplement and hashes |
| `scripts/ch10/train.py`, `pyproject.toml` | Optional ML dependencies, full fine-tune, best weight and validation threshold |
| `scripts/ch10/export_onnx.py` | Export with dynamic axes and full test-set threshold-label agreement |
| `scripts/ch10/inference_lib.py`, `serve.py`, `evaluate.py`, `scan_threshold_replay.py`, `classify_pool.py` | Shared inference semantics, 8110 API, held-out metrics and pool category report |
| `tests/ch10/`, `docs/ch10-topic-classifier-acceptance.md`, `dev-notes/ch10.md` | RED/GREEN, labeled Prompt check, repeatable acceptance and limits |

### Task 1: Authoritative taxonomy and pure corpus operations

**Files:** Create `app/core/taxonomy.py`, `scripts/ch10/__init__.py`, `scripts/ch10/corpus_lib.py`, `tests/ch10/test_taxonomy.py`, `tests/ch10/test_corpus_lib.py`.

**Interfaces:** `TopicClass(name, boundary, examples, severity)`; derived `TOPIC_NAMES`, `LABEL2ID`, `ID2LABEL`, `NUM_CLASSES`, `SEVERITY`, `terminology_table()`. `desensitize(text: str) -> str`, `dedupe(samples: list[dict]) -> list[dict]`, `split_dataset(samples: list[dict], seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]`.

- [ ] **Step 1:** Read source taxonomy and Context7-check any API imported for this unit. Write failing tests for exactly 17 ordered classes and derived maps; masking phone/email/order/social IDs while preserving model codes; stable dedupe and same-seed multilabel splits, including strata of 1–2 items.
- [ ] **Step 2:** Run `pytest tests/ch10/test_taxonomy.py tests/ch10/test_corpus_lib.py -q` and confirm RED. Implement the pure files with no network/DB imports; run GREEN.
- [ ] **Step 3:** Append Task 1 four-field journal entry and commit.

### Task 2: Private corpus, prelabel and human audit gate

**Files:** Create `scripts/ch10/build_corpus.py`, `scripts/ch10/prelabel.py`, `scripts/ch10/validate_golden.py`, `scripts/ch10/golden_samples.jsonl`, `tests/ch10/test_corpus.py`, `tests/ch10/test_prelabel.py`; modify `app/config.py` only for explicit privacy/job settings if required.

**Interfaces:** `load_pool(limit: int) -> list[dict]` returns source IDs and locally masked text; `prelabel_one(text: str, model, taxonomy) -> LabelResult` returns valid labels or explicit failure; `validate_golden(samples, predictions) -> GoldenReport`; `build_corpus(..., allow_external_real_text: bool = False) -> CorpusReport` writes versioned JSONL and audit Markdown only after privacy guard. A golden report below 0.8 prevents batch prelabel.

- [ ] **Step 1:** Context7-check LangChain `with_structured_output` and SQLAlchemy async reads. Create synthetic golden labels for all boundary classes; run the Prompt/result validator on those annotated examples. Add RED tests for external endpoint with no consent, ID masking, invalid label, upstream exception, low golden accuracy, per-class simulation target 100 (multilabel counted in each class), and fixed-seed audit sample of five per class.
- [ ] **Step 2:** Implement pool and model jobs; record `pending_upstream` or per-item failure without printing raw text or assigning “其他” to errors. Run focused tests GREEN; make a synthetic-only CLI run that does not contact paid services.
- [ ] **Step 3:** Append Task 2 four-field journal entry and commit.

### Task 3: Dataset split, augmentation and leakage checks

**Files:** Create `scripts/ch10/build_dataset.py`, `scripts/ch10/supplement_sizefit.jsonl`, `tests/ch10/test_dataset.py`.

**Interfaces:** `build_dataset(samples: list[dict], augment_fn=None, *, seed: int = 42) -> DatasetBuild` returns train/val/test and lineage hashes; `write_dataset(...)` persists only validated artifacts. An external augment model uses the same explicit privacy guard as Task 2.

- [ ] **Step 1:** RED tests: 80/10/10 stratification on sufficient combinations, rare-stratum behavior, same-seed repeatability, augment called only for train, duplicate paraphrase rejected across all splits, supplement only in train, and malformed/unknown labels rejected.
- [ ] **Step 2:** Implement minimal split→augment flow and cross-split normalized fingerprints. Run `pytest tests/ch10/test_dataset.py -q` GREEN and inspect emitted split manifest on synthetic data.
- [ ] **Step 3:** Append Task 3 four-field journal entry and commit.

### Task 4: Multilabel training and validation threshold

**Files:** Modify `pyproject.toml`; create `scripts/ch10/train.py`, `tests/ch10/test_train_logic.py`.

**Interfaces:** `encode(samples, tokenizer) -> list[dict]` creates 17-float labels; `scan_threshold(probs, gold) -> ThresholdResult` chooses first maximum micro-F1 from the exact grid; `train_from_dataset(...)` saves model/tokenizer/threshold plus taxonomy and dataset hashes or reports `pending_data`/`pending_compute`.

- [ ] **Step 1:** Context7-check installed Transformers `Trainer`/`TrainingArguments`, PyTorch model/export signatures and scikit-learn F1. Add optional `ml` dependency group. RED pure/fake tests for 17-float multi-hot, loss model configuration, early stopping/best-state copy, grid tie, and mismatched artifact hashes.
- [ ] **Step 2:** Implement lazy heavy imports, full-parameter fine-tune and one final model save. Run unit tests GREEN without model download; if local model weights and compute are unavailable, record `pending_compute` rather than a score.
- [ ] **Step 3:** Append Task 4 four-field journal entry and commit.

### Task 5: ONNX export and fresh-reference agreement

**Files:** Create `scripts/ch10/export_onnx.py`, `tests/ch10/test_export.py`.

**Interfaces:** `export_and_verify(model_dir, test_path, output_dir) -> ExportReport` uses TorchScript exporter `dynamo=False`, dynamic batch/seq axes, opset 17; reloads a fresh Torch reference after export; compares the complete test-set thresholded label matrix and always writes a report. Only zero mismatches is `passed`.

- [ ] **Step 1:** Context7-check `torch.onnx.export` and ONNX Runtime `InferenceSession.run`. RED tests with fake exporter/session for axes/opset, fresh reference reload, mismatch failure report and unavailable model status.
- [ ] **Step 2:** Implement export and verification, run focused tests GREEN. Do not download weights solely for a passing demo.
- [ ] **Step 3:** Append Task 5 four-field journal entry and commit.

### Task 6: Light serving, held-out evaluation and pool report

**Files:** Create `scripts/ch10/inference_lib.py`, `scripts/ch10/serve.py`, `scripts/ch10/evaluate.py`, `scripts/ch10/scan_threshold_replay.py`, `scripts/ch10/classify_pool.py`, `tests/ch10/test_inference.py`, `tests/ch10/test_serve.py`, `tests/ch10/test_evaluate.py`.

**Interfaces:** `apply_threshold(probs: ndarray, threshold: float) -> ndarray`; `load_runtime(model_dir) -> Runtime` verifies taxonomy hash and files before serving; `POST /classify` takes `texts: list[str]` and returns labels plus all 17 scores. Evaluation uses the same threshold function and reports micro/macro/per-class F1 by severity; replay scores validation once then scans fixed thresholds and compares saved threshold.

- [ ] **Step 1:** Context7-check FastAPI request models, tokenizers batching and ONNX Runtime session; RED tests for empty input, all-below-threshold fallback, label-order mismatch startup, score/label schema, held-out metrics with strict-class F1 0.9 and medium-class F1 0.8 redlines, strict-better tie and duplicate pool IDs.
- [ ] **Step 2:** Implement lazy runtime initialization so imports/tests do not require model files or Torch. Run focused tests GREEN and local `:8110` fake-runtime HTTP smoke; no new chat-path call.
- [ ] **Step 3:** Append Task 6 four-field journal entry and commit.

### Task 7: Offline acceptance and final independent review

**Files:** Create `docs/ch10-topic-classifier-acceptance.md`; modify `dev-notes/ch10.md` and tests only when review evidence requires.

**Interfaces:** Reproducible offline command sequence, dataset/golden/export/report statuses, precise missing prerequisites, and any measured real-model metrics only when actually run here.

- [ ] **Step 1:** Run `tests/ch10` and affected Chapter 9 tests with synthetic/isolated data; validate golden Prompt fixtures, CLI/HTTP fake-runtime path, hashes and privacy scan. Never present original MewHelp historical reports as this checkout's measurements.
- [ ] **Step 2:** Check actual model/data/quota/compute readiness. Run real training/export/`curl :8110/classify` only if prerequisites are present and the question scope permits; otherwise record `pending_data`, `pending_upstream` or `pending_compute` separately.
- [ ] **Step 3:** Use `superpowers:requesting-code-review` for whole-branch independent review; fix Important/Critical with `superpowers:receiving-code-review`, reverify, append review and finish four-field notes, commit.
