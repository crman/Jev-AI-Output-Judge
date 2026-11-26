# JevJudge — AI Response Evaluation Framework

JevJudge is a Python-based CLI evaluation framework for RAG-generated answers. It evaluates AI-generated responses against reference context using two judges, normalizes their outputs into a common verdict schema, measures evaluation quality against labelled ground truth, and exports structured reports.

---

## Judges

- **Jev** — a TypeSafe-hosted evaluation model that returns a continuous numeric support signal (`noul`).
- **Groq** — a Groq-hosted LLM used as a conventional judge with a structured prompt, returning a verdict and explanation.

---

## What it does

- Evaluates RAG answers for faithfulness to source context.
- Detects unsupported claims and contradictions.
- Normalizes provider-specific outputs into a common verdict schema.
- Measures judge performance against labelled ground truth using binary classification metrics.
- Compares the performance of different judges side by side.
- Exports evaluation reports in JSON and CSV format.

---

## Verdicts

| Verdict | Meaning |
|---|---|
| `supported` | The context supports the answer |
| `contradicted` | The context contradicts the answer |
| `insufficient_evidence` | The context does not provide enough information to evaluate the answer |
| `uncertain` | The result cannot be determined confidently |

---

## Project structure

```
app/
├── clients/
│   ├── groq_client.py          # Groq SDK wrapper
│   └── jev_client.py           # TypeSafe Jev HTTP client
├── evaluation/
│   ├── dataset_loader.py       # Load and validate the JSON dataset
│   ├── prompt_builder.py       # Build Groq system and user prompts
│   ├── jev_questions.py        # Jev question definition
│   ├── judge_normalizer.py     # Normalize raw judge responses
│   ├── pipeline.py             # Per-example and per-dataset evaluation
│   ├── metrics.py              # Binary classification metrics
│   ├── report_builder.py       # Build evaluation reports
│   ├── comparison_builder.py   # Build multi-judge comparison reports
│   └── report_exporter.py      # Export reports to JSON or CSV
├── schemas/
│   ├── evaluation.py           # Dataset, example, and ground truth models
│   ├── evaluation_result.py    # Verdict, judge type, and result models
│   ├── evaluation_report.py    # Single-judge report model
│   └── comparison_report.py    # Multi-judge comparison report model
├── cli.py                      # CLI entry point
└── config.py                   # Environment configuration

data/
└── evaluation_dataset.json     # 10-example labelled evaluation dataset

results/                        # Generated report files (gitignored)
tests/                          # Unit tests
docs/
└── JevJudge_Code_Walkthrough.md
```

---

## Getting started

### Prerequisites

- Python 3.11 or later
- A [Groq API key](https://console.groq.com/keys)
- A [TypeSafe API key](https://typesafe.ai/) for Jev

### 1. Clone the repository

```bash
git clone https://github.com/crman/jev-ai-output-judge.git
cd jev-ai-output-judge
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```
TYPESAFE_API_KEY=your-typesafe-api-key
JEV_MODEL=jev-latest
JEV_SUPPORT_THRESHOLD=0.5

GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=your-groq-model
```

`JEV_SUPPORT_THRESHOLD` is optional. When not set, all Jev results receive an `uncertain` verdict. See `app/evaluation/judge_normalizer.py` for the full threshold methodology.

---

## Running evaluations

### Jev evaluation

```bash
python -m app.cli --dataset data/evaluation_dataset.json --judge jev
```

### Groq evaluation

```bash
python -m app.cli --dataset data/evaluation_dataset.json --judge groq
```

### Export a JSON report

```bash
python -m app.cli \
  --dataset data/evaluation_dataset.json \
  --judge jev \
  --output results/jev_report.json \
  --format json
```

### Export a CSV report

```bash
python -m app.cli \
  --dataset data/evaluation_dataset.json \
  --judge groq \
  --output results/groq_report.csv \
  --format csv
```

### Multi-judge comparison

```bash
python -m app.cli \
  --dataset data/evaluation_dataset.json \
  --judge both \
  --output results/comparison.json \
  --format json
```

---

## Running tests

```bash
python -m pytest -q
```

All tests use mocking where live API calls would be needed. No test makes a real network request.

---

## Evaluation dataset

The dataset contains 10 labelled RAG evaluation examples across four categories:

| Category | Count | Description |
|---|---|---|
| `fully_supported` | 2 | Answer matches context |
| `contradiction` | 4 | Answer directly contradicts context |
| `unsupported_claim` | 3 | Answer adds claims not in context |
| `partially_supported` | 1 | Part of the answer is supported, part is not |

Each example has two independent ground truth labels: `contradiction` and `unsupported_claim`.

---

## Evaluation findings and judge comparison

Running both judges against the evaluation dataset highlights key architectural and behavioral differences between a continuous signal judge (Jev) and an LLM prompt-based judge (Groq).

### Benchmark results

| Example ID | Category | Ground truth | Jev verdict (threshold 0.5) | Groq verdict |
|---|---|---|---|---|
| rag_001 | fully_supported | Supported | supported (noul: 0.98) | supported |
| rag_002 | contradiction | Contradiction + Unsupported | insufficient_evidence (noul: 0.02) | contradicted |
| rag_003 | unsupported_claim | Unsupported claim | insufficient_evidence (noul: 0.07) | insufficient_evidence |
| rag_004 | partially_supported | Unsupported claim | insufficient_evidence (noul: 0.22) | supported |
| rag_005 | contradiction | Contradiction + Unsupported | insufficient_evidence (noul: 0.02) | contradicted |
| rag_006 | unsupported_claim | Unsupported claim | insufficient_evidence (noul: 0.39) | insufficient_evidence |
| rag_007 | fully_supported | Supported | supported (noul: 0.97) | supported |
| rag_008 | contradiction | Contradiction + Unsupported | insufficient_evidence (noul: 0.02) | contradicted |
| rag_009 | unsupported_claim | Unsupported claim | insufficient_evidence (noul: 0.06) | insufficient_evidence |
| rag_010 | contradiction | Contradiction + Unsupported | insufficient_evidence (noul: 0.02) | contradicted |

### Key observations

1. **Contradiction detection**
   - Groq accurately identifies direct contradictions (`contradicted` on rag_002, rag_005, rag_008, rag_010).
   - Jev's `hq_supported` question assesses evidence support rather than contradiction. Consequently, contradicted answers produce low support scores (`noul` <= 0.02) and map to `insufficient_evidence` under threshold rules.

2. **Partially supported statements**
   - On rag_004 (where valid delivery timeframe is mixed with an unsupported express delivery claim), Groq focused on the supported clause and yielded `supported`.
   - Jev registered the unsupported addition via a reduced support score (`noul: 0.22`), correctly classifying it as `insufficient_evidence`.

3. **Complementary evaluation roles**
   - Jev excels at continuous support measurement without prompt drift.
   - Groq provides qualitative categorical distinctions between contradictions and missing facts.

---

## Tech stack

| Component | Technology |
|---|---|
| Language | Python |
| Data validation | Pydantic |
| Configuration | Pydantic Settings |
| Jev client | httpx |
| Groq client | Groq Python SDK |
| Testing | pytest |

---

## Docs

- [Code Walkthrough](docs/JevJudge_Code_Walkthrough.md) — module-by-module explanation of the codebase
