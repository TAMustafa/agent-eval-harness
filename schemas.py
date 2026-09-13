from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SupportTriageTrace(BaseModel):
    """
    Data contract for customer support triage agent execution traces.
    Enforces deterministic syntax constraints and business-rule guardrails.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    action: Literal["issue_refund", "escalate_to_human", "reject_policy"]
    order_id: str | None = None
    refund_amount: float = Field(
        ge=0.0,
        description="Payout amount authorized for the refund. Must be >= 0.0.",
    )
    customer_reply: str = Field(
        min_length=1,
        description="Non-empty reply string generated for the customer.",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score bounded between 0.0 and 1.0.",
    )

    @model_validator(mode="after")
    def validate_business_rules(self) -> SupportTriageTrace:
        """
        Tier 1 Deterministic Guardrail Validators enforcing core business logic.
        """
        # Rule 1: A rejection must never authorize a payout
        if self.action == "reject_policy" and self.refund_amount > 0:
            raise ValueError(
                "A rejected refund cannot authorize a payout amount greater than zero."
            )

        # Rule 2: An authorized refund requires a traceable order ID
        if self.action == "issue_refund" and (self.order_id is None or not str(self.order_id).strip()):
            raise ValueError("Cannot issue a refund without a valid order_id.")

        # Rule 3: High-confidence actions should not waste human review resources
        if self.action == "escalate_to_human" and self.confidence_score >= 0.85:
            raise ValueError(
                "High confidence actions must not be routed to human escalation."
            )

        return self
