from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JudgeType(str, Enum):
    JEV = "jev"
    GROQ = "groq"


class EvaluationVerdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNCERTAIN = "uncertain"


class EvaluationResult(BaseModel):
    """Normalized evaluation result produced by an AI judge."""

    example_id: str = Field(min_length=1)
    judge: JudgeType
    model: str = Field(min_length=1)

    verdict: EvaluationVerdict
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    explanation: str | None = None

    # Preserve the original judge response for traceability.
    raw_output: dict[str, Any] | str

    # Optional token usage metadata, when available.
    usage: dict[str, Any] | None = None