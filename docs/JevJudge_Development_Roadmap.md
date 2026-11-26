# JevJudge — Complete Project Roadmap

**Project goal:** Build a reliable, modular AI evaluation framework for evaluating RAG-generated answers, detecting unsupported claims, measuring judge performance, and comparing AI judges.

This plan consolidates our project discussions, the work already completed, and the remaining implementation steps. It is intended to be our roadmap for continuing development, without adding unnecessary scope.

---

## 1. Project vision

JevJudge is a Python-based evaluation framework that evaluates AI-generated answers against reference context using multiple AI judges, normalizes their outputs into a common format, measures evaluation quality against labelled ground truth, and generates structured evaluation reports.

### Core objectives

- Evaluate RAG answers for support from provided context.
- Identify unsupported claims and contradictions through evaluation signals and verdicts.
- Integrate multiple evaluation providers.
- Normalize provider-specific outputs.
- Measure evaluation quality against labelled ground truth.
- Compare the performance of different judges.
- Generate machine-readable evaluation reports.
- Build a reusable, extensible Python architecture.

---

## 2. Current project status

### Completed and merged functionality

| Component | Status |
|---|---|
| Project structure and Python environment | Done |
| Configuration management | Done |
| TypeSafe Jev client | Done |
| Groq client | Done |
| Evaluation dataset and schema | Done |
| Dataset loader | Done |
| Groq evaluation prompts | Done |
| Jev evaluation questions | Done |
| Groq response normalization | Done |
| Jev response normalization | Done |
| Configurable Jev support threshold | Done |
| Evaluation pipeline | Done |
| CLI interface | Done |
| Verdict schema | Done |
| Binary classification metrics | Done |
| Evaluation metrics | Done |
| Multi-judge metrics comparison | Done |
| Evaluation report schema | Done |
| JSON report exporter | Done |
| CSV report exporter | Done |
| Evaluation report builder | Done |
| Comparison report schema/builder | Done |
| Unit tests for implemented components | Done |

### Pending integration

| Component | Status |
|---|---|
| CLI integration with report builder/exporter | Pending |
| CLI report output handling | Pending |
| End-to-end CLI JSON export | Pending |
| End-to-end CLI CSV export | Pending |
| CLI comparison workflow for both judges | Planned |
| Final validation of report presentation | Planned |
| Calibration methodology refinement | Planned |
| Final end-to-end validation | Planned |
| Portfolio cleanup | Planned |

---

## 3. Remaining development phases

### Phase 10 — Integrate report generation into the CLI

**Status:** Next development step

**Objective:** When an evaluation is executed from the CLI, the user should optionally be able to save the evaluation report in JSON or CSV format.

#### Tasks

- [ ] Inspect the current `app/cli.py`.
- [ ] Inspect the report builder and exporter interfaces.
- [ ] Integrate `build_evaluation_report()` into the CLI execution flow.
- [ ] Pass dataset metadata into the report builder.
- [ ] Ensure the report includes evaluation results and metrics.
- [ ] Wire `--output` to the report exporter.
- [ ] Wire `--format` to the JSON/CSV exporter.
- [ ] Handle invalid output formats and export errors appropriately.
- [ ] Preserve the existing CLI console summary.
- [ ] Add tests for report export through the CLI.
- [ ] Run the full test suite.

#### Expected usage

**JSON**

```bash
python -m app.cli \
  --dataset data/evaluation_dataset.json \
  --judge jev \
  --output results/jev_report.json \
  --format json
```

**CSV**

```bash
python -m app.cli \
  --dataset data/evaluation_dataset.json \
  --judge jev \
  --output results/jev_report.csv \
  --format csv
```

#### Expected outcome

- CLI evaluation works as it currently does.
- Reports are generated only when the user requests an output file.
- JSON and CSV reports contain normalized evaluation results.
- Report metrics are included where the format supports them.
- Errors are handled without unnecessarily crashing the CLI.

---

### Phase 11 — Multi-judge comparison through the CLI

**Status:** Planned

**Objective:** Run multiple judges against the same dataset and generate a unified comparison report.

Reuse the existing comparison infrastructure rather than reimplementing it.

#### Tasks

- [ ] Decide the CLI interface for running both judges.
- [ ] Extend CLI argument validation to support comparison mode.
- [ ] Execute Jev against the dataset.
- [ ] Execute Groq against the same dataset.
- [ ] Normalize results independently for each judge.
- [ ] Build individual evaluation reports for each judge.
- [ ] Build a comparison report using the existing comparison builder.
- [ ] Export the comparison report.
- [ ] Add tests for comparison-mode CLI execution.
- [ ] Run an actual comparison using both providers.

#### Proposed interface

```bash
python -m app.cli \
  --dataset data/evaluation_dataset.json \
  --judge both \
  --output results/comparison.json \
  --format json
```

> This is a proposed interface, not an implemented command. We will decide the exact CLI design when we reach this phase.

#### Expected outcome

A single execution should produce comparable results from both judges, including their individual evaluation metrics and a unified comparison.

---

### Phase 12 — Validate evaluation metrics and report presentation

**Status:** Planned

**Objective:** Ensure reports correctly represent judge performance and are useful for analysis.

#### Tasks

- [ ] Review accuracy.
- [ ] Review precision.
- [ ] Review recall.
- [ ] Review F1 score.
- [ ] Review contradiction detection metrics.
- [ ] Review unsupported-claim detection metrics.
- [ ] Verify how uncertain results are handled.
- [ ] Verify skipped results and failed evaluations.
- [ ] Verify metrics are calculated against the correct ground truth.
- [ ] Verify comparison metrics are consistent between judges.
- [ ] Inspect generated JSON and CSV files.

#### Expected outcome

The evaluation report should clearly communicate:

- Which judge was used.
- Which model was used, where applicable.
- Which dataset was evaluated.
- Individual verdicts.
- Metrics.
- Errors or skipped results.
- Raw signals necessary for further analysis.

> This phase is primarily validation. We should only change metric implementation if we identify an actual issue.

---

### Phase 13 — Jev threshold and calibration methodology

**Status:** Planned

**Objective:** Make the threshold-based interpretation of Jev's support signal transparent and properly documented.

#### Current implementation

- Jev returns a `noul` support signal.
- The project supports a configurable `JEV_SUPPORT_THRESHOLD`.
- A score meeting the threshold is mapped to `SUPPORTED`.
- A score below the threshold is mapped to `INSUFFICIENT_EVIDENCE`.
- Without a threshold, the verdict is `UNCERTAIN`.

#### Tasks

- [ ] Review the current threshold implementation.
- [ ] Keep the threshold configurable.
- [ ] Document that threshold-based verdicts are project-defined interpretations.
- [ ] Distinguish the raw Jev signal from the normalized verdict.
- [ ] Review the exploratory calibration results.
- [ ] Avoid presenting the exploratory threshold as independently validated.
- [ ] Consider a separate calibration dataset in the future.

#### Important distinction

- **Jev's raw signal:** The signal returned by the evaluation provider.
- **Project threshold:** The configurable value used to interpret that signal.
- **Normalized verdict:** The verdict assigned by the framework using its interpretation rules.

Jev's current question evaluates support. Therefore, a low support score alone does not establish that an answer contradicts the context.

---

### Phase 14 — Code quality and test review

**Status:** Planned

**Objective:** Review the codebase for correctness, consistency, and maintainability before final validation.

#### Tasks

- [ ] Review module responsibilities.
- [ ] Review error handling.
- [ ] Review type hints.
- [ ] Review duplicated code.
- [ ] Review client resource cleanup.
- [ ] Review configuration handling.
- [ ] Review test coverage for edge cases.
- [ ] Check for unused imports.
- [ ] Review CLI error messages.
- [ ] Verify `.env.example`.
- [ ] Verify secrets are not included in Git.
- [ ] Run the complete test suite.
- [ ] Run `git diff --check`.

#### Commands

```bash
python -m pytest -q
```

```bash
git diff --check
```

> We will not undertake broad refactoring unless we find something worth fixing.

---

### Phase 15 — End-to-end validation

**Status:** Planned

**Objective:** Validate the project as a real user would use it, including live provider calls and generated output files.

#### Validation scenarios

##### 1. Jev evaluation

```bash
python -m app.cli \
  --dataset data/evaluation_dataset.json \
  --judge jev
```

##### 2. Jev JSON report

```bash
python -m app.cli \
  --dataset data/evaluation_dataset.json \
  --judge jev \
  --output results/jev_report.json \
  --format json
```

##### 3. Jev CSV report

```bash
python -m app.cli \
  --dataset data/evaluation_dataset.json \
  --judge jev \
  --output results/jev_report.csv \
  --format csv
```

##### 4. Groq evaluation

```bash
python -m app.cli \
  --dataset data/evaluation_dataset.json \
  --judge groq
```

##### 5. Groq JSON report

```bash
python -m app.cli \
  --dataset data/evaluation_dataset.json \
  --judge groq \
  --output results/groq_report.json \
  --format json
```

##### 6. Multi-judge comparison

After Phase 11 has been implemented, validate the comparison command and inspect the resulting comparison report.

#### Validation checklist

- [ ] Both providers execute successfully.
- [ ] Results are normalized correctly.
- [ ] Verdicts are consistent with the defined interpretation.
- [ ] Metrics are calculated correctly.
- [ ] JSON files are valid.
- [ ] CSV files contain the expected columns.
- [ ] Reports contain the expected data.
- [ ] Errors are reported properly.
- [ ] Output files are generated in the expected location.

---

### Phase 16 — Final project cleanup and portfolio readiness

**Status:** Final phase

**Objective:** Make JevJudge ready for GitHub portfolio presentation.

#### Tasks

- [ ] Review the final project structure.
- [ ] Update README only where necessary.
- [ ] Ensure setup instructions work from a fresh clone.
- [ ] Ensure `.env.example` is accurate.
- [ ] Remove unnecessary temporary files.
- [ ] Review sample results, if included.
- [ ] Ensure secrets are not committed.
- [ ] Verify tests.
- [ ] Verify CLI commands.
- [ ] Review Git history.
- [ ] Verify the repository is easy to understand.

#### Expected outcome

**JevJudge — Portfolio Ready**

A modular, tested, CLI-driven AI evaluation framework supporting multiple judges, normalized evaluation results, metrics, reporting, and judge comparison.

---

## 4. Development workflow

We will continue using the existing Git workflow for each feature.

### Step 1 — Update main

```bash
git switch main
git pull origin main
```

### Step 2 — Create a feature branch

```bash
git switch -c feat/feature-name
```

### Step 3 — Implement the feature

Make focused changes, following the existing architecture.

### Step 4 — Add tests

Test new functionality and relevant edge cases.

### Step 5 — Run tests

```bash
python -m pytest -q
```

### Step 6 — Review changes

```bash
git status
git diff --check
git diff
```

### Step 7 — Commit and push

Commit only the intended files and push the feature branch.

### Step 8 — Create a PR

Review the changes before merging into `main`.

> We will continue one implementation step at a time. We will not jump ahead, merge automatically, or modify unrelated files.

---

## 5. Final roadmap overview

| Phase | Work | Status |
|---|---|---|
| 1–9 | Core architecture, judges, pipeline, metrics, reports | Completed |
| 10 | CLI report export integration | **Next** |
| 11 | Multi-judge CLI comparison | Planned |
| 12 | Metrics and report validation | Planned |
| 13 | Jev threshold/calibration methodology | Planned |
| 14 | Code quality and test review | Planned |
| 15 | End-to-end validation | Planned |
| 16 | Portfolio cleanup | Planned |

---

## Our next task: Phase 10

Integrate the existing report builders and exporters into the CLI so that evaluation runs can generate JSON and CSV reports through the command-line interface.

**No new architecture, no unnecessary refactoring, and no unrelated README changes.**
