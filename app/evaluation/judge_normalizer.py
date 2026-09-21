import json
import re
from math import isfinite
from typing import Any

from app.schemas.evaluation_result import (
    EvaluationResult,
    EvaluationVerdict,
    JudgeType,
)


class JudgeNormalizationError(ValueError):
    """Raised when a judge response cannot be normalized."""


def normalize_groq_response(
    example_id: str,
    model: str,
    raw_output: str,
) -> EvaluationResult:
    """
    Convert a Groq judge response into the common EvaluationResult schema.

    Expected response format:
    {
        "verdict": "supported",
        "explanation": "The claim is supported by the evidence."
    }
    """

    if not raw_output or not raw_output.strip():
        raise JudgeNormalizationError("Groq response is empty.")

    # Extract JSON from the response.
    try:
        parsed: Any = json.loads(raw_output)
    except json.JSONDecodeError:
        # Support JSON wrapped in Markdown code fences.
        match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            raw_output,
            re.DOTALL | re.IGNORECASE,
        )

        if not match:
            raise JudgeNormalizationError(
                "Groq response does not contain valid JSON."
            )

        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise JudgeNormalizationError(
                "Groq response contains malformed JSON."
            ) from exc

    if not isinstance(parsed, dict):
        raise JudgeNormalizationError(
            "Groq response must be a JSON object."
        )

    verdict_value = parsed.get("verdict")

    if not isinstance(verdict_value, str):
        raise JudgeNormalizationError(
            "Groq response is missing a valid verdict."
        )

    try:
        verdict = EvaluationVerdict(verdict_value.lower().strip())
    except ValueError as exc:
        raise JudgeNormalizationError(
            f"Unsupported Groq verdict: {verdict_value}"
        ) from exc

    explanation = parsed.get("explanation")

    if explanation is not None and not isinstance(explanation, str):
        raise JudgeNormalizationError(
            "Groq explanation must be a string."
        )

    return EvaluationResult(
        example_id=example_id,
        judge=JudgeType.GROQ,
        model=model,
        verdict=verdict,
        explanation=explanation,
        raw_output=raw_output,
    )
    
    
def normalize_jev_response(
    example_id: str,
    raw_output: dict[str, Any],
) -> EvaluationResult:
    """
    Normalize a Jev API response into the common EvaluationResult schema.

    Jev's noul signal is preserved, but is not mapped to a definitive
    verdict until its semantics and decision threshold are established.
    """

    if not isinstance(raw_output, dict):
        raise JudgeNormalizationError(
            "Jev response must be a dictionary."
        )

    model = raw_output.get("model")
    if not isinstance(model, str) or not model.strip():
        raise JudgeNormalizationError(
            "Jev response is missing a valid model."
        )

    answers = raw_output.get("answers")
    if not isinstance(answers, dict):
        raise JudgeNormalizationError(
            "Jev response is missing answers."
        )

    hq_supported = answers.get("hq_supported")
    if not isinstance(hq_supported, dict):
        raise JudgeNormalizationError(
            "Jev response is missing hq_supported."
        )

    noul_value = hq_supported.get("noul")

    if (
        not isinstance(noul_value, (int, float))
        or isinstance(noul_value, bool)
        or not isfinite(noul_value)
        or not 0.0 <= noul_value <= 1.0
    ):
        raise JudgeNormalizationError(
            "Jev noul value must be a finite number between 0 and 1."
        )

    usage = raw_output.get("usage")

    return EvaluationResult(
        example_id=example_id,
        judge=JudgeType.JEV,
        model=model,
        verdict=EvaluationVerdict.UNCERTAIN,
        confidence=None,
        explanation=(
            "Jev returned an hq_supported noul signal. "
            "No verdict threshold has been configured."
        ),
        raw_output=raw_output,
        usage=usage if isinstance(usage, dict) else None,
    )