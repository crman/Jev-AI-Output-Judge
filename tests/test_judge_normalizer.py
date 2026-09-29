import pytest

from app.evaluation.judge_normalizer import (
    JudgeNormalizationError,
    normalize_groq_response,
    normalize_jev_response,
)
from app.schemas.evaluation_result import (
    EvaluationVerdict,
    JudgeType,
)


def test_normalize_groq_valid_response():
    raw_output = (
        '{"verdict": "supported", '
        '"explanation": "The evidence supports the claim."}'
    )

    result = normalize_groq_response(
        example_id="rag_001",
        model="test-model",
        raw_output=raw_output,
    )

    assert result.example_id == "rag_001"
    assert result.judge == JudgeType.GROQ
    assert result.model == "test-model"
    assert result.verdict == EvaluationVerdict.SUPPORTED
    assert result.explanation == "The evidence supports the claim."
    assert result.raw_output == raw_output


def test_normalize_groq_markdown_json():
    raw_output = """```json
    {
        "verdict": "contradicted",
        "explanation": "The evidence contradicts the claim."
    }
    ```"""

    result = normalize_groq_response(
        example_id="rag_002",
        model="test-model",
        raw_output=raw_output,
    )

    assert result.verdict == EvaluationVerdict.CONTRADICTED


def test_normalize_groq_invalid_json():
    with pytest.raises(JudgeNormalizationError):
        normalize_groq_response(
            example_id="rag_003",
            model="test-model",
            raw_output="This is not JSON",
        )


def test_normalize_groq_invalid_verdict():
    raw_output = '{"verdict": "maybe", "explanation": "Unsure"}'

    with pytest.raises(JudgeNormalizationError):
        normalize_groq_response(
            example_id="rag_004",
            model="test-model",
            raw_output=raw_output,
        )


def test_normalize_groq_empty_response():
    with pytest.raises(JudgeNormalizationError):
        normalize_groq_response(
            example_id="rag_005",
            model="test-model",
            raw_output="",
        )
        
def test_normalize_jev_valid_response():
    raw_output = {
        "model": "jev-1.13.0",
        "answers": {
            "hq_supported": {
                "type": "noul",
                "noul": 0.05,
            }
        },
        "usage": {
            "input_tokens": 305,
            "output_tokens": 21,
        },
    }

    result = normalize_jev_response(
        example_id="rag_001",
        raw_output=raw_output,
    )

    assert result.example_id == "rag_001"
    assert result.judge == JudgeType.JEV
    assert result.model == "jev-1.13.0"
    assert result.verdict == EvaluationVerdict.UNCERTAIN
    assert result.confidence is None
    assert result.raw_output == raw_output
    assert result.usage == raw_output["usage"]


def test_normalize_jev_missing_answers():
    with pytest.raises(JudgeNormalizationError):
        normalize_jev_response(
            example_id="rag_001",
            raw_output={"model": "jev-1.13.0"},
        )


@pytest.mark.parametrize("noul", [-0.1, 1.1, "0.5", True, None])
def test_normalize_jev_invalid_noul(noul):
    raw_output = {
        "model": "jev-1.13.0",
        "answers": {
            "hq_supported": {
                "type": "noul",
                "noul": noul,
            }
        },
    }

    with pytest.raises(JudgeNormalizationError):
        normalize_jev_response(
            example_id="rag_001",
            raw_output=raw_output,
        )
        
def test_normalize_jev_with_support_threshold_returns_supported():
    raw_output = {
        "model": "jev-1.13.0",
        "answers": {
            "hq_supported": {
                "type": "noul",
                "noul": 0.90,
            }
        },
    }

    result = normalize_jev_response(
        example_id="rag_001",
        raw_output=raw_output,
        support_threshold=0.50,
    )

    assert result.verdict == EvaluationVerdict.SUPPORTED
    assert result.example_id == "rag_001"


def test_normalize_jev_with_support_threshold_returns_insufficient_evidence():
    raw_output = {
        "model": "jev-1.13.0",
        "answers": {
            "hq_supported": {
                "type": "noul",
                "noul": 0.20,
            }
        },
    }

    result = normalize_jev_response(
        example_id="rag_002",
        raw_output=raw_output,
        support_threshold=0.50,
    )

    assert result.verdict == EvaluationVerdict.INSUFFICIENT_EVIDENCE


def test_normalize_jev_without_threshold_returns_uncertain():
    raw_output = {
        "model": "jev-1.13.0",
        "answers": {
            "hq_supported": {
                "type": "noul",
                "noul": 0.90,
            }
        },
    }

    result = normalize_jev_response(
        example_id="rag_001",
        raw_output=raw_output,
    )

    assert result.verdict == EvaluationVerdict.UNCERTAIN


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1],
)
def test_normalize_jev_rejects_invalid_support_threshold(threshold):
    raw_output = {
        "model": "jev-1.13.0",
        "answers": {
            "hq_supported": {
                "type": "noul",
                "noul": 0.50,
            }
        },
    }

    with pytest.raises(
        JudgeNormalizationError,
        match="support threshold must be between 0 and 1",
    ):
        normalize_jev_response(
            example_id="rag_001",
            raw_output=raw_output,
            support_threshold=threshold,
        )