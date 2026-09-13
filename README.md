# Offline Agent Evaluation Harness

A clean, production-ready test harness demonstrating how to test LLM agents **without instantiating or running the agent during test time**. 

This repository accompanies the technical guide on [tarek-ai.com](https://tarek-ai.com) explaining the two-tiered agent evaluation strategy:
1. **Tier 1 (Deterministic Guardrails):** Instantaneous, zero-cost syntax and business-rule validation using Pydantic.
2. **Tier 2 (Probabilistic Semantic Evals):** LLM-as-a-judge evaluation of Faithfulness (hallucination detection) and Answer Relevancy using DeepEval and Pytest, with automatic fast-failing to minimize token expenditure.

---

## Architecture Overview

```
                          Simulated Agent Execution Trace
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │    Tier 1: Pydantic Guardrails    │
                     │  - Action / Field Validation      │
                     │  - Business Invariant Enforcement │
                     └─────────────────┬─────────────────┘
                                       │
                         Passes Tier 1 │ Fails Tier 1
                                       │ ──────────────┐
                                       ▼               ▼
                     ┌───────────────────────────┐  [ Fast-Fail ]
                     │  Tier 2: DeepEval Semantic│  Skip Tier 2 to
                     │  - Faithfulness (> 0.70)  │  save LLM judge
                     │  - Relevancy    (> 0.70)  │  tokens
                     └───────────────────────────┘
```

### Why Decouple Agent Execution from Evaluation?
Running an autonomous agent end-to-end inside test suites introduces:
- **Flakiness:** Non-deterministic tool selection and network calls.
- **Extreme Latency:** Multi-step reasoning loops take 15–45 seconds per test.
- **Uncontrolled Cost:** Continuous integration runs burn hundreds of LLM calls per PR.

By persisting **execution traces** (e.g. from production logs or sandbox simulation) into a structured dataset (`dataset/traces.json`), evaluation becomes deterministic, repeatable, and fast.

---

## Directory Structure

```text
agent-eval-harness/
├── .env.example            # Environment configuration template
├── README.md               # Setup and architecture documentation
├── requirements.txt        # Pydantic, DeepEval, Pytest, python-dotenv
├── dataset/
│   └── traces.json         # 5 production traces (passing and targeted failures)
├── schemas.py              # Pydantic data contract & business rule validators
└── test_evals.py           # Two-tiered Pytest test suite
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (recommended for fast virtualenv and package management) or standard Python `venv`

### 2. Setup Virtual Environment with `uv`

Navigate into the project directory:
```bash
cd agent-eval-harness
```

Create a virtual environment with Python 3.11+ using `uv`:
```bash
uv venv --python 3.11
source .venv/bin/activate
```

Install the dependencies:
```bash
uv pip install -r requirements.txt
```

*(Alternatively, with standard pip: `pip install -r requirements.txt`)*

### 3. Environment Configuration

Copy the example environment file and set your DeepSeek API key (used with `deepseek-chat` for Tier 2 LLM-as-a-judge evaluations):

```bash
cp .env.example .env
```

Edit `.env` and provide your API key:
```env
DEEPSEEK_API_KEY=sk-...
```

> **Note:** Tier 1 tests run **100% offline** and do not require an API key or internet access. Tier 2 tests use DeepSeek's cheapest model (`deepseek-chat`, DeepSeek-V3) at ~$0.14-$0.28 / 1M tokens, and will gracefully skip if `DEEPSEEK_API_KEY` is not present.

---

## Running the Evaluations

### Method 1: Run via Standard Pytest (Recommended)

Run both Tier 1 and Tier 2 tests with verbose output:
```bash
pytest test_evals.py -v
```

Run **only** Tier 1 deterministic guardrails (instant & offline):
```bash
pytest test_evals.py -k "test_tier1" -v
```

Run **only** Tier 2 semantic evaluations:
```bash
pytest test_evals.py -k "test_tier2" -v
```

### Method 2: Run via DeepEval CLI

DeepEval provides an interactive terminal UI with scoring summaries:
```bash
deepeval test run test_evals.py
```

---

## Evaluation Test Cases in `dataset/traces.json`

| Trace ID | Expected Tier 1 Result | Expected Tier 2 Result | What It Demonstrates |
| :--- | :--- | :--- | :--- |
| `pass_valid_refund` | **PASS** | **PASS** | Valid damaged item refund authorized within 30 days. |
| `pass_valid_rejection` | **PASS** | **PASS** | Proper policy rejection for 6-month-old electronics. |
| `fail_pydantic_schema_business_rule` | **FAIL** (ValidationError) | **SKIPPED** (Fast-Fail) | Catches bug: rejection action authorizing a payout amount ($30.00). |
| `fail_deepeval_hallucination` | **PASS** | **FAIL** (Faithfulness < 0.7) | Catches fabricated 180-day worldwide policy contradiction. |
| `fail_deepeval_relevancy` | **PASS** | **FAIL** (Relevancy < 0.7) | Catches response evading question about opened toner returns. |

---

## Two-Tiered Architecture Details

### Tier 1: Pydantic Business-Rule Guardrails (`schemas.py`)
Deterministic data contracts validate the structure and domain rules using `@model_validator(mode="after")`:
- **Payout on Rejection:** If `action == "reject_policy"` and `refund_amount > 0`, reject the trace immediately.
- **Refund Traceability:** If `action == "issue_refund"` and `order_id` is missing or empty, reject the trace.
- **Human Routing Economics:** If `action == "escalate_to_human"` and `confidence_score >= 0.85`, reject to prevent unnecessary manual escalations.

### Tier 2: Probabilistic Semantic Guardrails (`test_evals.py`)
Constructs an `LLMTestCase` evaluated by:
- **`FaithfulnessMetric(threshold=0.7)`:** Checks whether the agent's output can be inferred directly from the `retrieval_context` without hallucinations.
- **`AnswerRelevancyMetric(threshold=0.7)`:** Quantifies how directly the agent's response answers the customer's question.
- **Fast-Fail Token Optimization:** Before running Tier 2, traces are validated against `SupportTriageTrace`. If Tier 1 fails, Tier 2 skips immediately, saving cost and time.
