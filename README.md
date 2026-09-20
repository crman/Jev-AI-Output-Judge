# JevJudge — AI Response Evaluation & Hallucination Detection

> Exploring Jev as a structured decision-making model for evaluating LLM-generated responses.

JevJudge is an AI engineering project focused on evaluating the quality and reliability of LLM-generated answers using [Jev](https://typesafe.ai/) and LLMs served through [Groq](https://groq.com/).

The goal is to investigate whether Jev can serve as an effective evaluation layer for Retrieval-Augmented Generation (RAG) applications.

This project will evolve incrementally, starting with a small evaluation dataset and expanding into a reusable benchmarking application.

## 🎯 Objectives

* Explore Jev's structured decision-making capabilities.
* Evaluate LLM-generated answers against source context.
* Detect contradictions and unsupported claims.
* Compare Jev with a Groq-hosted LLM judge.
* Measure evaluation accuracy, latency, and cost.
* Build a reproducible evaluation pipeline using Python.

## 🏗️ Planned Architecture

```text
          Evaluation Dataset
                  |
                  v
          Evaluation Runner
                  |
          +-------+-------+
          |               |
          v               v
       Jev Judge      Groq LLM Judge
          |               |
          v               v
      Evaluation       Evaluation
       Results          Results
          |               |
          +-------+-------+
                  |
                  v
           Metrics Engine
                  |
                  v
         Benchmark Reports
```

## 🛠️ Tech Stack

| Component            | Technology         |
| -------------------- | ------------------ |
| Programming Language | Python             |
| Decision Model       | Jev                |
| LLM Provider         | Groq               |
| API Integration      | Jev API / Groq SDK |
| Data Validation      | Pydantic           |
| Testing              | Pytest             |
| Version Control      | Git & GitHub       |

Additional tools, such as Streamlit, may be introduced as the project evolves.

## 🧪 Evaluation Criteria

The initial benchmark will focus on:

* **Faithfulness:** Is the answer supported by the provided context?
* **Contradiction Detection:** Does the answer conflict with the source?
* **Unsupported Claims:** Does the answer introduce claims that the context does not support?
* **Accuracy:** How closely do model judgments match human-verified labels?
* **Latency:** How long does each model take to evaluate an example?
* **Cost:** What is the estimated cost of evaluating a given number of examples?

## 📂 Project Structure

```text
jev-ai-output-judge/
├── app/
│   ├── clients/
│   ├── evaluation/
│   ├── schemas/
│   ├── config.py
│   └── main.py
├── data/
├── tests/
├── results/
├── docs/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

* Python 3.11+
* Git
* A [Groq API key](https://console.groq.com/keys)
* Jev API access (required for Jev evaluation)

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

Add your API credentials and selected model identifiers to `.env`.

Never commit API keys or other secrets to GitHub.

### 5. Run the application

Application execution instructions will be added as the evaluation pipeline is implemented.

## 📊 Current Status

**Stage:** Initial project setup

The repository structure, Groq configuration, and initial documentation are in place.

Dataset creation and model integration are upcoming milestones.

No benchmark results or model performance claims have been established yet.

## 🤝 Learning & Experimentation

This repository is an ongoing hands-on exploration of structured AI decision-making, LLM evaluation, and reliable AI engineering.

The implementation, experiments, and findings will be documented as the project progresses.

Contributions, feedback, and discussions are welcome.
