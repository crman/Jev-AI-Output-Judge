import pytest

from app.evaluation.comparison_builder import build_comparison_report
from app.schemas.comparison_report import ComparisonReport
from app.schemas.evaluation_report import EvaluationReport
from app.schemas.evaluation_result import (
    EvaluationResult,
    EvaluationVerdict,
    JudgeType,
)


def make_report(
    judge: JudgeType,
    model: str,
    dataset_name: str = "test_dataset",
    dataset_version: str = "1.0",
):
    result = EvaluationResult(
        example_id="rag_001",
        judge=judge,
        model=model,
        verdict=EvaluationVerdict.SUPPORTED,
        raw_output="{}",
    )

    return EvaluationReport(
        report_id=f"{judge.value}_report",
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        judge=judge,
        model=model,
        results=[result],
        metrics={
            "contradiction": {
                "accuracy": 1.0,
            }
        },
        evaluated_count=1,
        skipped_count=0,
    )


def test_build_comparison_report():
    groq_report = make_report(
        JudgeType.GROQ,
        "test-groq",
    )

    jev_report = make_report(
        JudgeType.JEV,
        "test-jev",
    )

    comparison = build_comparison_report(
        [
            groq_report,
            jev_report,
        ]
    )

    assert isinstance(
        comparison,
        ComparisonReport,
    )

    assert comparison.dataset_name == "test_dataset"
    assert comparison.dataset_version == "1.0"

    assert len(comparison.reports) == 2

    assert comparison.reports[0].judge == JudgeType.GROQ
    assert comparison.reports[1].judge == JudgeType.JEV

    assert comparison.comparison_id


def test_comparison_requires_reports():
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        build_comparison_report([])


def test_comparison_rejects_different_datasets():
    groq_report = make_report(
        JudgeType.GROQ,
        "test-groq",
        dataset_name="dataset_a",
    )

    jev_report = make_report(
        JudgeType.JEV,
        "test-jev",
        dataset_name="dataset_b",
    )

    with pytest.raises(
        ValueError,
        match="same dataset",
    ):
        build_comparison_report(
            [
                groq_report,
                jev_report,
            ]
        )


def test_comparison_rejects_different_dataset_versions():
    groq_report = make_report(
        JudgeType.GROQ,
        "test-groq",
        dataset_version="1.0",
    )

    jev_report = make_report(
        JudgeType.JEV,
        "test-jev",
        dataset_version="2.0",
    )

    with pytest.raises(
        ValueError,
        match="same dataset version",
    ):
        build_comparison_report(
            [
                groq_report,
                jev_report,
            ]
        )


def test_comparison_rejects_duplicate_judges():
    first_groq = make_report(
        JudgeType.GROQ,
        "model-a",
    )

    second_groq = make_report(
        JudgeType.GROQ,
        "model-b",
    )

    with pytest.raises(
        ValueError,
        match="duplicate judges",
    ):
        build_comparison_report(
            [
                first_groq,
                second_groq,
            ]
        )