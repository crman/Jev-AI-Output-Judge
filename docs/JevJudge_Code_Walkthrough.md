# JevJudge — Code Walkthrough

This document explains every module in the project: what it does, how it fits into the overall architecture, and the decisions made during implementation. It is updated whenever new changes are made.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Architecture map](#2-architecture-map)
3. [Configuration — `app/config.py`](#3-configuration--appconfigpy)
4. [Schemas — `app/schemas/`](#4-schemas--appschemas)
5. [Clients — `app/clients/`](#5-clients--appclients)
6. [Evaluation — `app/evaluation/`](#6-evaluation--appevaluation)
7. [CLI — `app/cli.py`](#7-cli--appclipy)
8. [Data — `data/`](#8-data--data)
9. [Tests — `tests/`](#9-tests--tests)
10. [Change log](#10-change-log)

---

## 1. Project overview

JevJudge is a Python-based CLI evaluation framework for RAG (Retrieval-Augmented Generation) answers. It evaluates AI-generated answers against reference context using two AI judges:

- **Jev** — a TypeSafe-hosted evaluation model that returns a numeric support signal (`noul`).
- **Groq** — a Groq-hosted LLM used as an evaluation judge with a custom structured prompt.

The framework normalizes provider-specific outputs into a common verdict schema, measures evaluation quality against labelled ground truth, and exports structured reports in JSON or CSV format.

---

## 2. Architecture map

```
app/
├── config.py                   ← Environment settings (API keys, model names, threshold)
├── cli.py                      ← CLI entry point and orchestration
├── clients/
│   ├── groq_client.py          ← HTTP client wrapping the Groq SDK
│   └── jev_client.py           ← HTTP client for the TypeSafe Jev API
├── evaluation/
│   ├── dataset_loader.py       ← Load and validate the JSON dataset
│   ├── prompt_builder.py       ← Build Groq system + user prompts
│   ├── jev_questions.py        ← Jev question definition (hq_supported)
│   ├── judge_normalizer.py     ← Normalize raw judge responses → EvaluationResult
│   ├── pipeline.py             ← Per-example and per-dataset evaluation runners
│   ├── metrics.py              ← Binary classification metrics + multi-judge comparison
│   ├── report_builder.py       ← Build EvaluationReport from results
│   ├── comparison_builder.py   ← Build ComparisonReport from multiple EvaluationReports
│   └── report_exporter.py      ← Export reports to JSON or CSV files
└── schemas/
    ├── evaluation.py           ← EvaluationDataset, EvaluationExample, GroundTruth
    ├── evaluation_result.py    ← EvaluationResult, EvaluationVerdict, JudgeType
    ├── evaluation_report.py    ← EvaluationReport (single-judge report)
    └── comparison_report.py    ← ComparisonReport (multi-judge report)

data/
└── evaluation_dataset.json     ← 10-example labelled RAG evaluation dataset

tests/                          ← Unit tests mirroring the app/ structure
docs/
├── JevJudge_Development_Roadmap.md
└── JevJudge_Code_Walkthrough.md   ← this file
```

---

## 3. Configuration — `app/config.py`

**File:** [`app/config.py`](../app/config.py)

Uses `pydantic-settings` to load environment variables from `.env`.

```python
class Settings(BaseSettings):
    typesafe_api_key: str = ""   # API key for TypeSafe / Jev
    jev_model: str = "jev-latest"
    jev_support_threshold: float | None = None  # Optional; None → UNCERTAIN verdict

    groq_api_key: str = ""
    groq_model: str = ""
```

**Key design decisions:**

- `jev_support_threshold` defaults to `None`. When not set, Jev results produce an `UNCERTAIN` verdict instead of being mapped to `SUPPORTED` or `INSUFFICIENT_EVIDENCE`. This keeps the threshold explicitly opt-in.
- A single `settings` singleton is instantiated at module level and imported throughout the project.

---

## 4. Schemas — `app/schemas/`

Schemas are Pydantic models. They are the shared language of the whole system — every module speaks through these types.

### 4.1 `evaluation.py` — Dataset types

| Class | Purpose |
|---|---|
| `GroundTruth` | Two boolean labels per example: `contradiction`, `unsupported_claim` |
| `EvaluationExample` | One dataset row: `id`, `question`, `context`, `answer`, `ground_truth`, `category` |
| `EvaluationDataset` | The full dataset: `dataset_name`, `version`, `examples` |

`GroundTruth` has two independent labels because a single answer can be both a contradiction *and* an unsupported claim (or neither). They are not mutually exclusive.

### 4.2 `evaluation_result.py` — Judge output

| Class / Enum | Purpose |
|---|---|
| `JudgeType` | `"jev"` or `"groq"` — identifies the provider |
| `EvaluationVerdict` | `supported`, `contradicted`, `insufficient_evidence`, `uncertain` |
| `EvaluationResult` | Normalized output from one judge for one example |

`EvaluationResult` fields:

| Field | Type | Notes |
|---|---|---|
| `example_id` | `str` | Links back to the dataset example |
| `judge` | `JudgeType` | Which provider produced this result |
| `model` | `str` | Model name used (from the provider response) |
| `verdict` | `EvaluationVerdict` | Normalized verdict |
| `confidence` | `float \| None` | Optional; `0.0–1.0`; not currently used by either provider |
| `explanation` | `str \| None` | Groq provides a string; Jev gets a generated explanation |
| `raw_output` | `dict \| str` | Preserved original response for traceability |
| `usage` | `dict \| None` | Token usage, when available |

### 4.3 `evaluation_report.py` — Single-judge report

```python
class EvaluationReport(BaseModel):
    report_id: str           # UUID
    dataset_name: str
    dataset_version: str
    judge: JudgeType
    model: str
    created_at: datetime
    results: list[EvaluationResult]
    metrics: dict            # {"contradiction": {...}, "unsupported_claim": {...}}
    errors: list[dict]
    evaluated_count: int     # Excludes UNCERTAIN results
    skipped_count: int       # Count of UNCERTAIN results excluded from metrics
```

`metrics` is stored as a plain `dict` (using `vars()` on the dataclass) for straightforward JSON serialisation.

### 4.4 `comparison_report.py` — Multi-judge report

```python
class ComparisonReport(BaseModel):
    comparison_id: str       # UUID
    dataset_name: str
    dataset_version: str
    created_at: datetime
    reports: list[EvaluationReport]   # One per judge
```

A `ComparisonReport` is a thin wrapper around multiple `EvaluationReport` objects validated against the same dataset. It does not recompute metrics — it embeds the full reports side by side.

---

## 5. Clients — `app/clients/`

### 5.1 `groq_client.py` — GroqClient

**File:** [`app/clients/groq_client.py`](../app/clients/groq_client.py)

Wraps the official `groq` Python SDK.

```python
client.generate(system_prompt, user_prompt) → str
```

- Uses `temperature=0.0` to get deterministic responses.
- Returns the raw assistant message text — JSON parsing happens in the normalizer, not here.
- `close()` shuts down the underlying HTTP client. The CLI calls this in a `finally` block.

**Validation on init:** Raises `ValueError` if `api_key` or `model` are empty, so misconfiguration fails early.

### 5.2 `jev_client.py` — JevClient

**File:** [`app/clients/jev_client.py`](../app/clients/jev_client.py)

A thin `httpx`-based HTTP client for the TypeSafe Jev API.

```
POST https://api.typesafe.ai/v1/systemone
```

Request payload:
```json
{
  "model": "jev-latest",
  "state": { "question": "...", "context": "...", "answer": "..." },
  "questions": { "hq_supported": { "type": "noul", "instructions": "..." } }
}
```

Returns the parsed JSON response dict. Raises `httpx.HTTPStatusError` on HTTP errors, `httpx.RequestError` on transport failures. The client does not interpret the response — that is the normalizer's job.

---

## 6. Evaluation — `app/evaluation/`

This is the core of the framework. The modules form a linear pipeline:

```
dataset_loader → pipeline → judge_normalizer → metrics → report_builder → report_exporter
```

### 6.1 `dataset_loader.py`

**File:** [`app/evaluation/dataset_loader.py`](../app/evaluation/dataset_loader.py)

```python
load_dataset(file_path) → EvaluationDataset
```

Reads a JSON file and validates it through `EvaluationDataset.model_validate()`. Raises `FileNotFoundError` if the path does not exist. Validation errors propagate from Pydantic.

### 6.2 `prompt_builder.py`

**File:** [`app/evaluation/prompt_builder.py`](../app/evaluation/prompt_builder.py)

```python
build_groq_prompts(example) → (system_prompt, user_prompt)
```

Builds the two Groq prompts for one example:

- **System prompt:** Instructs the model to act as an impartial evaluator and return a JSON object with `verdict` and `explanation`. Explicitly asks for no Markdown fences.
- **User prompt:** The `question`, `context`, and `answer` as a JSON string.

Ground-truth labels are **not included** in the prompt — the judge must decide independently.

### 6.3 `jev_questions.py`

**File:** [`app/evaluation/jev_questions.py`](../app/evaluation/jev_questions.py)

Defines the single Jev question used for evaluation:

```python
JEV_EVALUATION_QUESTIONS = {
    "hq_supported": {
        "type": "noul",
        "instructions": "Is the answer supported by the provided context?"
    }
}
```

`noul` is Jev's continuous numeric signal type (0.0–1.0). The question is named `hq_supported` and is reused across every example in the dataset.

### 6.4 `judge_normalizer.py`

**File:** [`app/evaluation/judge_normalizer.py`](../app/evaluation/judge_normalizer.py)

The normalizer is responsible for converting raw provider responses into the common `EvaluationResult` schema. It raises `JudgeNormalizationError` (a `ValueError` subclass) on any invalid or unexpected response shape.

#### Groq normalization — `normalize_groq_response()`

Input: raw assistant message text (`str`).

Steps:
1. Try to parse the string as JSON directly.
2. If that fails, look for a JSON block inside Markdown code fences (` ```json ... ``` `).
3. Validate that the parsed object has a `"verdict"` string.
4. Map the verdict string to `EvaluationVerdict`.

Groq's `confidence` field is not populated because the prompt does not ask the model to provide one.

#### Jev normalization — `normalize_jev_response()`

Input: raw parsed JSON dict from the Jev API.

Steps:
1. Extract `model`, `answers.hq_supported.noul`.
2. Validate that `noul` is a finite float between 0.0 and 1.0.
3. Apply threshold logic:
   - No threshold → `UNCERTAIN` verdict.
   - `noul >= threshold` → `SUPPORTED`.
   - `noul < threshold` → `INSUFFICIENT_EVIDENCE`.

#### Jev threshold methodology

Three concepts are kept distinct throughout the codebase:

| Concept | What it is |
|---|---|
| **Raw Jev signal** | The `noul` value returned by the Jev API — always preserved in `raw_output` |
| **Project threshold** | The configurable `JEV_SUPPORT_THRESHOLD` — a project-defined decision boundary, not provided or validated by TypeSafe |
| **Normalized verdict** | The `EvaluationVerdict` the framework assigns using the threshold — the framework's interpretation, not an independent Jev judgment |

> **Important:** Jev's question only evaluates *support*. A low support score means the answer is not well-supported by the context — it does not mean the answer *contradicts* the context. The `CONTRADICTED` verdict is never assigned by the Jev normalizer. Only Groq can produce it.

> **Calibration note:** The threshold used in this project is exploratory and has not been independently validated on a held-out calibration dataset. If this framework is used in production, the threshold should be calibrated against a labelled dataset that is separate from the evaluation dataset.

### 6.5 `pipeline.py`

**File:** [`app/evaluation/pipeline.py`](../app/evaluation/pipeline.py)

The pipeline orchestrates evaluation from dataset to raw results. It has three layers:

| Function | Purpose |
|---|---|
| `evaluate_groq_example()` | Call client + normalize for one Groq example |
| `evaluate_jev_example()` | Call client + normalize for one Jev example |
| `evaluate_example()` | Dispatch to the right provider, validate required arguments |
| `evaluate_dataset()` | Iterate over examples; catch per-example errors; return `{results, errors}` |
| `evaluate_groq_dataset()` | Build Groq prompts, then call `evaluate_dataset()` |
| `evaluate_jev_dataset()` | Build Jev state dict, then call `evaluate_dataset()` |
| `run_dataset_evaluation()` | Top-level entry: load dataset, choose provider, run evaluation |

**Error handling:** `evaluate_dataset()` wraps each example in a `try/except`. A failing example is captured as an error dict `{example_id, error_type, message}` and does not stop the rest of the evaluation.

### 6.6 `metrics.py`

**File:** [`app/evaluation/metrics.py`](../app/evaluation/metrics.py)

Computes binary classification metrics against labelled ground truth.

#### Data classes

| Class | Fields |
|---|---|
| `BinaryClassificationMetrics` | `accuracy`, `precision`, `recall`, `f1`, `true_positive`, `true_negative`, `false_positive`, `false_negative` |
| `EvaluationMetrics` | `contradiction` (BCM), `unsupported_claim` (BCM), `evaluated_count`, `skipped_count` |
| `JudgeEvaluationMetrics` | `judge`, `model`, `metrics` (EvaluationMetrics) |
| `JudgeComparisonMetrics` | `judges` (list of JudgeEvaluationMetrics) |

#### How verdicts map to labels — `verdict_to_labels()`

| Verdict | contradiction label | unsupported_claim label |
|---|---|---|
| `SUPPORTED` | `False` | `False` |
| `CONTRADICTED` | `True` | `False` |
| `INSUFFICIENT_EVIDENCE` | `False` | `True` |
| `UNCERTAIN` | `None` | `None` |

`None` means "skip this result". Results with `UNCERTAIN` verdicts are excluded from metric calculation and counted in `skipped_count`.

#### Metric formulas

$$\text{accuracy} = \frac{TP + TN}{N}$$

$$\text{precision} = \frac{TP}{TP + FP}$$

$$\text{recall} = \frac{TP}{TP + FN}$$

$$\text{F1} = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

Division-by-zero is handled gracefully — returns `0.0` rather than raising.

#### Multi-judge comparison — `calculate_judge_comparison()`

Accepts a `dict[JudgeType, Sequence[EvaluationResult]]` and calculates `JudgeEvaluationMetrics` for each judge independently against the same ground truth.

### 6.7 `report_builder.py`

**File:** [`app/evaluation/report_builder.py`](../app/evaluation/report_builder.py)

```python
build_evaluation_report(dataset, judge, model, results, errors) → EvaluationReport
```

Validates that:
- `results` is non-empty.
- All results match the given `judge` and `model`.
- There are no duplicate `example_id` values.

Then builds a `ground_truth` lookup dict from the dataset and calls `calculate_judge_metrics()` to get the metrics. Returns a fully populated `EvaluationReport` with a UUID `report_id`.

### 6.8 `comparison_builder.py`

**File:** [`app/evaluation/comparison_builder.py`](../app/evaluation/comparison_builder.py)

```python
build_comparison_report(reports) → ComparisonReport
```

Validates that:
- `reports` is non-empty.
- All reports use the same `dataset_name` and `dataset_version`.
- No duplicate judges.

Returns a `ComparisonReport` wrapping the list of `EvaluationReport` objects with a new UUID `comparison_id`.

### 6.9 `report_exporter.py`

**File:** [`app/evaluation/report_exporter.py`](../app/evaluation/report_exporter.py)

Exports either an `EvaluationReport` or a `ComparisonReport` to disk.

#### JSON export — `export_report_to_json()`

Calls `report.model_dump(mode="json")` (Pydantic) and writes pretty-printed JSON with `indent=2`. Parent directories are created automatically.

#### CSV export — `export_report_to_csv()`

Writes one row per `EvaluationResult`. For a `ComparisonReport`, results from all sub-reports are flattened into a single file. The `judge` column in each row identifies which judge produced it.

CSV columns: `example_id`, `judge`, `model`, `verdict`, `confidence`, `explanation`, `raw_output`.

> **Note:** The JSON report includes metrics; the CSV does not — it only contains per-result rows.

#### Convenience wrappers

`export_comparison_to_json()` and `export_comparison_to_csv()` are thin wrappers that delegate to the main functions. They exist for callers that want an explicitly typed interface.

---

## 7. CLI — `app/cli.py`

**File:** [`app/cli.py`](../app/cli.py)

The CLI is the user-facing entry point. It is also the orchestrator — it calls every other module in the right order.

### Arguments

| Argument | Required | Values | Purpose |
|---|---|---|---|
| `--dataset` | Yes | Path to JSON | Dataset to evaluate |
| `--judge` | Yes | `jev`, `groq`, `both` | Which judge(s) to run |
| `--output` | No | File path | Where to write the report |
| `--format` | No | `json`, `csv` | Report format (default: `json`) |

### Internal helpers

| Function | Purpose |
|---|---|
| `_run_groq_evaluation(dataset_path)` | Creates `GroqClient`, runs dataset evaluation, closes client |
| `_run_jev_evaluation(dataset_path)` | Creates `JevClient`, runs dataset evaluation, closes client |
| `_print_judge_summary(judge, data)` | Prints result count, errors, and Jev signals to stdout |
| `_build_judge_report(dataset, judge, data)` | Calls `build_evaluation_report()`; returns `None` if no results |

### Execution flow

#### Single judge (`--judge jev` or `--judge groq`)

```
load_dataset()
  → _run_[judge]_evaluation()
    → run_dataset_evaluation()
  → _print_judge_summary()
  → _build_judge_report()
  → [if --output] export_report_to_json() or export_report_to_csv()
```

#### Multi-judge (`--judge both`)

```
load_dataset()
  → _run_jev_evaluation()   → _print_judge_summary(JEV)
  → _run_groq_evaluation()  → _print_judge_summary(GROQ)
  → _build_judge_report(JEV)
  → _build_judge_report(GROQ)
  → build_comparison_report([jev_report, groq_report])
  → [if --output] export_report_to_json() or export_report_to_csv()
```

### Error handling

| Situation | Behaviour |
|---|---|
| No results produced and `--output` given | Raises `ValueError("Cannot export report: no evaluation results were produced.")` |
| One judge fails in `--judge both` mode | Raises `ValueError("Cannot build comparison report: ...")` |
| Per-example evaluation failure | Captured as an error dict; evaluation continues |

### Usage examples

```bash
# Jev evaluation, no report file
python -m app.cli --dataset data/evaluation_dataset.json --judge jev

# Groq evaluation, JSON report
python -m app.cli --dataset data/evaluation_dataset.json --judge groq \
  --output results/groq_report.json --format json

# Multi-judge comparison, CSV
python -m app.cli --dataset data/evaluation_dataset.json --judge both \
  --output results/comparison.csv --format csv
```

---

## 8. Data — `data/`

### `evaluation_dataset.json`

10 hand-labelled RAG evaluation examples. Each has:
- A `question` and `context` (the reference).
- An `answer` to evaluate.
- `ground_truth` with two independent boolean labels.
- A `category` for human readability.

| category | Count | Description |
|---|---|---|
| `fully_supported` | 2 | Answer matches the context exactly |
| `contradiction` | 4 | Answer directly contradicts the context |
| `unsupported_claim` | 3 | Answer adds claims not present in the context |
| `partially_supported` | 1 | Part of the answer is supported, part is not |

The dataset is intentionally small but covers all meaningful verdict scenarios. It is the shared benchmark both judges are evaluated against.

---

## 9. Tests — `tests/`

All tests use `pytest`. Each test module mirrors a corresponding `app/` module.

| Test file | What it covers |
|---|---|
| `test_cli.py` | CLI argument parsing, single-judge and multi-judge export, error cases |
| `test_comparison_builder.py` | ComparisonReport validation and error cases |
| `test_dataset_loader.py` | JSON loading, missing file, invalid structure |
| `test_evaluation_pipeline.py` | Per-example and per-dataset evaluation, error capture |
| `test_evaluation_report.py` | EvaluationReport schema validation |
| `test_evaluation_result.py` | EvaluationResult field validation |
| `test_groq_client.py` | GroqClient construction, missing credentials |
| `test_jev_client.py` | JevClient construction, missing credentials |
| `test_judge_normalizer.py` | Groq and Jev normalization, edge cases, invalid inputs |
| `test_metrics.py` | Binary metrics, verdict mapping, multi-judge comparison |
| `test_prompt_builder.py` | Prompt structure, ground-truth exclusion |
| `test_report_builder.py` | Report build, duplicate ID detection, metric integration |
| `test_report_exporter.py` | JSON and CSV file generation and content validation |

All tests use mocking (`monkeypatch`, `unittest.mock`) where live API calls would be needed. No test makes a real network request.

---

## 10. Change log

| Phase | What changed | Date |
|---|---|---|
| 1–9 | Core architecture: config, clients, dataset, pipeline, normalizers, metrics, schemas, report builder/exporter, CLI (single-judge) | Before roadmap |
| 10 | CLI: wired `--output` and `--format` to report exporters for single-judge runs | 2026-11-26 |
| 11 | CLI: added `--judge both` mode; refactored evaluation helpers; `report_exporter` extended to accept `ComparisonReport`; full test coverage for both phases | 2026-11-26 |
| 12 | Phase 12: removed unreachable `return` in `export_comparison_to_csv`; added this walkthrough document | 2026-11-26 |
| 13 | Phase 13: added module-level methodology note to `judge_normalizer.py` distinguishing raw signal, project threshold, and normalized verdict; expanded threshold comments in `config.py`; updated walkthrough | 2026-11-26 |
| 14 | Code quality: removed duplicate `run_cli()` call in `cli.py`; normalised UTC import order in `evaluation_report.py` | 2026-11-26 |
