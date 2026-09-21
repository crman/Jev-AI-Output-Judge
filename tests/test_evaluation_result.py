import pytest
from pydantic import ValidationError

from app.schemas.evaluation_result import (
    EvaluationResult,
    EvaluationVerdict,
    JudgeType,
)


def test_create_valid_evaluation_result():
    result = EvaluationResult(
        example_id="rag_006",
        judge="jev",
        model="jev-1.13.0",
        verdict="insufficient_evidence",
        confidence=0.95,
        explanation="The context does not establish the headquarters.",
        raw_output={
            "hq_supported": {
                "type": "noul",
                "noul": 0.05,
            }
        },
    )

    assert result.example_id == "rag_006"
    assert result.judge == JudgeType.JEV
    assert result.verdict == EvaluationVerdict.INSUFFICIENT_EVIDENCE
    assert result.confidence == 0.95


def test_confidence_is_optional():
    result = EvaluationResult(
        example_id="rag_001",
        judge="groq",
        model="openai/gpt-oss-120b",
        verdict="supported",
        raw_output="The answer is supported by the context.",
    )

    assert result.confidence is None


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        EvaluationResult(
            example_id="rag_001",
            judge="jev",
            model="jev-1.13.0",
            verdict="supported",
            confidence=1.5,
            raw_output={"answer": 0.9},
        )


def test_invalid_verdict_is_rejected():
    with pytest.raises(ValidationError):
        EvaluationResult(
            example_id="rag_001",
            judge="groq",
            model="openai/gpt-oss-120b",
            verdict="maybe",
            raw_output="Maybe supported.",
        )