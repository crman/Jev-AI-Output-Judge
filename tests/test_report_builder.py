
import pytest

from app.evaluation.report_builder import build_evaluation_report
from app.schemas.evaluation import (
    EvaluationDataset,
    EvaluationExample,
    GroundTruth,
)
from app.schemas.evaluation_result import (
    EvaluationResult,
    EvaluationVerdict,
    JudgeType,
)


def make_dataset():
    return EvaluationDataset(
        dataset_name="test_dataset",
        version="1.0",
        examples=[
            EvaluationExample(
                id="rag_001",
                question="What is the capital of France?",
                context="Paris is the capital of France.",
                answer="Paris is the capital of France.",
                ground_truth=GroundTruth(
                    contradiction=False,
                    unsupported_claim=False,
                ),
                category="factual",
            ),
            EvaluationExample(
                id="rag_002",
                question="What is the capital of Germany?",
                context="Berlin is the capital of Germany.",
                answer="Munich is the capital of Germany.",
                ground_truth=GroundTruth(
                    contradiction=True,
                    unsupported_claim=False,
                ),
                category="factual",
            ),
        ],
    )


def make_result(example_id, verdict, model="test-model"):
    return EvaluationResult(
        example_id=example_id,
        judge=JudgeType.GROQ,
        model=model,
        verdict=verdict,
        raw_output="{}",
    )


def test_build_evaluation_report():
    dataset = make_dataset()

    results = [
        make_result("rag_001", EvaluationVerdict.SUPPORTED),
        make_result("rag_002", EvaluationVerdict.CONTRADICTED),
    ]

    report = build_evaluation_report(
        dataset=dataset,
        judge=JudgeType.GROQ,
        model="test-model",
        results=results,
    )

    assert report.dataset_name == "test_dataset"
    assert report.dataset_version == "1.0"
    assert report.judge == JudgeType.GROQ
    assert report.model == "test-model"

    assert len(report.results) == 2
    assert report.evaluated_count == 2
    assert report.skipped_count == 0

    assert report.metrics["contradiction"]["accuracy"] == 1.0
    assert report.metrics["unsupported_claim"]["accuracy"] == 1.0

    assert report.errors == []
    assert report.report_id


def test_report_tracks_uncertain_results():
    dataset = make_dataset()

    results = [
        make_result("rag_001", EvaluationVerdict.SUPPORTED),
        make_result("rag_002", EvaluationVerdict.UNCERTAIN),
    ]

    report = build_evaluation_report(
        dataset,
        JudgeType.GROQ,
        "test-model",
        results,
    )

    assert len(report.results) == 2
    assert report.evaluated_count == 1
    assert report.skipped_count == 1


def test_report_preserves_pipeline_errors():
    dataset = make_dataset()

    errors = [
        {
            "example_id": "rag_002",
            "error": "API timeout",
        }
    ]

    report = build_evaluation_report(
        dataset=dataset,
        judge=JudgeType.GROQ,
        model="test-model",
        results=[
            make_result(
                "rag_001",
                EvaluationVerdict.SUPPORTED,
            )
        ],
        errors=errors,
    )

    assert report.evaluated_count == 1
    assert report.errors == errors


def test_report_rejects_mismatched_model():
    dataset = make_dataset()

    with pytest.raises(ValueError, match="judge and model"):
        build_evaluation_report(
            dataset,
            JudgeType.GROQ,
            "expected-model",
            [
                make_result(
                    "rag_001",
                    EvaluationVerdict.SUPPORTED,
                    model="different-model",
                )
            ],
        )


def test_report_requires_results():
    with pytest.raises(ValueError, match="At least one"):
        build_evaluation_report(
            make_dataset(),
            JudgeType.GROQ,
            "test-model",
            [],
        )


def test_report_rejects_duplicate_example_ids():
    result = make_result(
        "rag_001",
        EvaluationVerdict.SUPPORTED,
    )

    with pytest.raises(ValueError, match="Duplicate example IDs"):
        build_evaluation_report(
            make_dataset(),
            JudgeType.GROQ,
            "test-model",
            [result, result],
        )