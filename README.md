# JevJudge — AI Response Evaluation & Hallucination Detection

> A hands-on AI engineering project exploring Jev and LLM-based judges for evaluating the quality, faithfulness, and reliability of generated responses.

**JevJudge** is an evaluation framework for testing LLM-generated responses against source context and ground-truth labels.

The project currently evaluates responses using two judge approaches:

- **[Jev](https://typesafe.ai/)** — a structured decision-making model accessed through the Jev API
- **[Groq](https://groq.com/)** — an LLM inference platform used to run a conventional LLM-based evaluation judge

The goal is to investigate how different judge approaches perform when evaluating responses generated from Retrieval-Augmented Generation (RAG) scenarios.

The project is being built incrementally, with an emphasis on reproducible evaluation, clear separation between judge execution and metric calculation, and testable Python components.

---

## 🎯 Objectives

* Explore Jev's structured decision-making capabilities.
* Evaluate LLM-generated answers against source context.
* Detect contradictions and unsupported claims.
* Compare Jev with a Groq-hosted LLM judge.
* Build a reproducible evaluation pipeline using Python.
* Measure judge performance against ground-truth labels.
* Handle uncertain evaluation results explicitly rather than silently treating them as correct or incorrect.
* Build a reusable foundation for future benchmarking and reporting.

---

## 🏗️ Current Architecture

```text
                  Evaluation Dataset
                         │
                         ▼
                 Dataset Loader
                         │
                         ▼
                Evaluation Pipeline
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
         Groq Judge             Jev Judge
              │                     │
              ▼                     ▼
       Response Normalization
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                Evaluation Results
                         │
                         ▼
                  Metrics Engine
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Groq Metrics            Jev Metrics
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                 Comparison Data
```

The architecture intentionally separates:

1. **Dataset loading**
2. **Judge execution**
3. **Result normalization**
4. **Metric calculation**
5. **Future reporting and benchmarking**

This allows individual components to be tested independently.

---

## 🛠️ Tech Stack

| Component            | Technology         |
| -------------------- | ------------------ |
| Programming Language | Python             |
| Decision Model       | Jev                |
| LLM Provider         | Groq               |
| API Integration      | Jev API / Groq SDK |
| Data Validation      | Pydantic           |
| Testing              | Pytest             |
| Configuration        | Pydantic Settings  |
| Version Control      | Git & GitHub       |

---

## 🧪 Evaluation Dataset

The project currently contains a small synthetic evaluation dataset designed around RAG-style response evaluation.

Each example contains:

* Question
* Source context
* Generated answer
* Ground-truth labels
* Evaluation category

Example structure:

```json
{
  "id": "rag_001",
  "question": "What is the capital of France?",
  "context": "Paris is the capital of France.",
  "answer": "Paris is the capital of France.",
  "ground_truth": {
    "contradiction": false,
    "unsupported_claim": false
  },
  "category": "factual"
}
```

Ground-truth labels currently capture two independent properties:

* `contradiction`
* `unsupported_claim`

The ground truth is used only for measuring judge performance and is not included in judge prompts.

---

## ⚖️ Evaluation Verdicts

The evaluation pipeline normalizes judge responses into a common set of verdicts:

| Verdict                 | Meaning                                                             |
| ----------------------- | ------------------------------------------------------------------- |
| `supported`             | The answer is supported by the provided context.                    |
| `contradicted`          | The answer conflicts with the provided context.                     |
| `insufficient_evidence` | The context does not provide enough evidence to support the answer. |
| `uncertain`             | The judge cannot produce a sufficiently confident evaluation.       |

Both Jev and Groq results are converted into the common `EvaluationResult` schema.

This provides a consistent representation regardless of which judge produced the result.

---

## 🔄 Evaluation Pipeline

The evaluation pipeline supports both single-example and dataset-level evaluation.

### Single-example evaluation

```text
Question + Context + Answer
            │
            ▼
       Selected Judge
            │
            ▼
     Raw Judge Output
            │
            ▼
    Result Normalization
            │
            ▼
     EvaluationResult
```

### Dataset evaluation

```text
Evaluation Dataset
        │
        ▼
   Dataset Loader
        │
        ▼
   Selected Judge
        │
        ▼
  Batch Evaluation
        │
        ├── Successful Results
        │
        └── Per-example Errors
```

The batch pipeline continues evaluating remaining examples when an individual example fails, allowing evaluation runs to produce partial results instead of terminating immediately.

---

## 📐 Evaluation Metrics

The project currently includes binary classification metrics for evaluating judge predictions against ground truth.

### Metrics

* Accuracy
* Precision
* Recall
* F1 score

### Confusion-matrix counts

* True Positive
* True Negative
* False Positive
* False Negative

Metrics are calculated independently for:

* `contradiction`
* `unsupported_claim`

### Uncertain results

`UNCERTAIN` judge results are explicitly excluded from metric calculations rather than being automatically treated as negative predictions.

The metrics also track:

* `evaluated_count`
* `skipped_count`

This keeps the reported metrics transparent about how many results were actually evaluated.

---

## 🔬 Judge Comparison

The metrics layer supports calculating evaluation metrics independently for multiple judges.

For example:

```text
                    Ground Truth
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
         Groq Judge              Jev Judge
             │                       │
             ▼                       ▼
        Groq Metrics             Jev Metrics
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                  Comparison Data
```

The current implementation reports metrics for each judge independently. It does not automatically rank or select a preferred judge.

---

## 📂 Project Structure

```text
jev-ai-output-judge/
│
├── app/
│   ├── clients/
│   │   ├── groq_client.py
│   │   └── jev_client.py
│   │
│   ├── evaluation/
│   │   ├── dataset_loader.py
│   │   ├── judge_normalizer.py
│   │   ├── metrics.py
│   │   ├── pipeline.py
│   │   └── prompt_builder.py
│   │
│   ├── schemas/
│   │   ├── evaluation.py
│   │   └── evaluation_result.py
│   │
│   ├── config.py
│   └── main.py
│
├── data/
│   └── evaluation_dataset.json
│
├── tests/
│   ├── test_dataset_loader.py
│   ├── test_evaluation_pipeline.py
│   ├── test_metrics.py
│   ├── test_prompt_builder.py
│   └── ...
│
├── results/
├── docs/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.11+
* Git
* A [Groq API key](https://console.groq.com/keys)
* Jev API access for Jev evaluation

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

Add the required API credentials and model identifiers to `.env`.

Example configuration:

```text
TYPESAFE_API_KEY=<your-jev-api-key>
JEV_MODEL=<jev-model>

GROQ_API_KEY=<your-groq-api-key>
GROQ_MODEL=<groq-model>
```

**Never commit API keys or other secrets to GitHub.**

---

## 🧪 Running Tests

Run the complete test suite:

```bash
python -m pytest -v
```

Run a specific test module:

```bash
python -m pytest tests/test_metrics.py -v
```

The project follows an incremental test-driven development approach, with tests covering:

* Dataset validation
* Judge clients
* Result normalization
* Prompt generation
* Single-example evaluation
* Batch evaluation
* Error handling
* Evaluation metrics
* Judge comparison

---


## 🤝 Learning & Experimentation

JevJudge is an ongoing hands-on exploration of:

* LLM-as-a-Judge systems
* Structured AI decision-making
* RAG evaluation
* Hallucination detection
* Evaluation metrics
* Reliable AI engineering
* Reproducible benchmarking

The project is intentionally being developed incrementally, with each capability backed by automated tests.

Future experiments will focus on comparing different judge approaches across the same evaluation dataset and documenting the resulting observations.

This repository is an ongoing hands-on exploration of structured AI decision-making, LLM evaluation, and reliable AI engineering.

The implementation, experiments, and findings will be documented as the project progresses.

Contributions, feedback, and discussions are welcome.

---

## 📌 Project Philosophy

The project focuses on **measurement rather than assumptions**.

Instead of assuming that one evaluation approach is better than another, JevJudge aims to create a reproducible framework where different judges can be evaluated against the same dataset, ground truth, and metrics.

This makes the project useful not only as a Jev experiment, but also as a foundation for exploring broader LLM evaluation techniques.
