"""
Agent Evaluation Harness: Two-Tiered Evaluation Suite

Tier 1: Deterministic syntax & business-rule guardrails using Pydantic.
Tier 2: Probabilistic semantic evaluation (hallucination & relevancy) using DeepEval.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import ValidationError

# Opt out of interactive DeepEval telemetry prompts during automated runs
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

load_dotenv()

from schemas import SupportTriageTrace

DATASET_PATH = Path(__file__).parent / "dataset" / "traces.json"


def load_traces() -> list[dict]:
    """Load simulated execution traces from dataset/traces.json."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


TRACES = load_traces()


@pytest.fixture(scope="session")
def traces() -> list[dict]:
    """Pytest session fixture returning the full trace dataset."""
    return TRACES


# ============================================================================
# Tier 1: Deterministic Syntax & Business-Rule Guardrails (Pydantic)
# ============================================================================


@pytest.mark.parametrize("trace", TRACES, ids=[t["id"] for t in TRACES])
def test_tier1_deterministic_guardrails(trace: dict):
    """
    Tier 1 Evaluation:
    Asserts schema compliance, field ranges, and strict business invariants
    without making any LLM calls (zero latency, zero API cost).

    Covers all three business-rule guardrails defined in schemas.py:
      Rule 1 — A rejection must never authorise a non-zero payout.
      Rule 2 — An authorised refund requires a traceable order_id.
      Rule 3 — High-confidence actions must not be routed to human escalation.
    """
    raw_output = trace["agent_raw_output"]

    if trace["id"] == "fail_pydantic_schema_business_rule":
        # Rule 1: reject_policy action with a non-zero payout amount
        with pytest.raises(ValidationError) as exc_info:
            SupportTriageTrace(**raw_output)

        assert "A rejected refund cannot authorize a payout amount greater than zero." in str(
            exc_info.value
        )

    elif trace["id"] == "fail_pydantic_refund_missing_order_id":
        # Rule 2: issue_refund action with no order_id — payout is untraceable
        with pytest.raises(ValidationError) as exc_info:
            SupportTriageTrace(**raw_output)

        assert "Cannot issue a refund without a valid order_id." in str(exc_info.value)

    elif trace["id"] == "fail_pydantic_high_confidence_escalation":
        # Rule 3: escalate_to_human action with confidence_score >= 0.85
        with pytest.raises(ValidationError) as exc_info:
            SupportTriageTrace(**raw_output)

        assert "High confidence actions must not be routed to human escalation." in str(
            exc_info.value
        )

    else:
        # All other traces must strictly adhere to the SupportTriageTrace contract
        validated = SupportTriageTrace(**raw_output)
        assert validated is not None
        assert validated.action == raw_output["action"]
        assert validated.refund_amount == raw_output["refund_amount"]


# ============================================================================
# Tier 2: Probabilistic Semantic Evaluation (DeepEval + LLM-as-a-Judge)
# ============================================================================


@pytest.mark.parametrize("trace", TRACES, ids=[t["id"] for t in TRACES])
def test_tier2_semantic_evals(trace: dict):
    """
    Tier 2 Evaluation:
    Evaluates semantic quality (faithfulness/hallucination and answer relevancy)
    using DeepEval LLM-as-a-judge metrics.

    Includes fast-fail optimization: traces that fail Tier 1 are skipped
    to prevent wasteful LLM token expenditure.
    """
    # Fast-fail optimization: ensure Tier 1 passes before invoking LLM judge
    try:
        SupportTriageTrace(**trace["agent_raw_output"])
    except ValidationError:
        pytest.skip(
            f"Fast-fail cost optimization: Trace '{trace['id']}' failed Tier 1 guardrails. "
            "Skipping Tier 2 semantic evaluation to avoid burning LLM judge tokens."
        )

    # Check for DeepSeek API Key before running probabilistic evaluations
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip(
            "DEEPSEEK_API_KEY not configured. Set DEEPSEEK_API_KEY in your environment or .env file to run Tier 2 evaluations."
        )

    # Ensure DEEPSEEK_API_KEY is available in the environment for DeepEval internals
    os.environ.setdefault("DEEPSEEK_API_KEY", api_key)

    # Lazy import DeepEval components to ensure Tier 1 can run in lightweight environments
    try:
        from deepeval import assert_test
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.models import DeepSeekModel
        from deepeval.test_case import LLMTestCase
    except ImportError as e:
        pytest.fail(f"DeepEval is not installed. Install requirements via `uv pip install -r requirements.txt`: {e}")

    # Initialize DeepSeek's cheapest model: deepseek-chat (DeepSeek-V3)
    judge_model = DeepSeekModel(model="deepseek-chat", api_key=api_key)

    # Construct the DeepEval LLM test case
    test_case = LLMTestCase(
        input=trace["customer_input"],
        actual_output=trace["agent_raw_output"]["customer_reply"],
        retrieval_context=trace["retrieved_policy"],
    )

    # Define semantic evaluation metrics with 0.7 quality threshold using deepseek-chat
    faithfulness = FaithfulnessMetric(threshold=0.7, model=judge_model)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge_model)

    if trace["id"] == "fail_deepeval_hallucination":
        # The agent fabricated a 180-day worldwide policy not in retrieval context.
        # Faithfulness metric must drop below 0.7 and fail the assertion.
        with pytest.raises(AssertionError):
            assert_test(test_case, [faithfulness])

    elif trace["id"] == "fail_deepeval_relevancy":
        # The agent replied with shipping info instead of addressing the toner return.
        # Relevancy metric must drop below 0.7 and fail the assertion.
        with pytest.raises(AssertionError):
            assert_test(test_case, [relevancy])

    else:
        # High-quality production traces must pass both faithfulness and relevancy
        assert_test(test_case, [faithfulness, relevancy])
