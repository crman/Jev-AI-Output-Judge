from datetime import datetime

from app.schemas.evaluation_report import EvaluationReport
from app.schemas.evaluation_result import (
    EvaluationResult,
    EvaluationVerdict,
    JudgeType,
)


def test_evaluation_report_creation():
    result = EvaluationResult(
        example_id="rag_001",
        judge=JudgeType.GROQ,
        model="test-model",
        verdict=EvaluationVerdict.SUPPORTED,
        raw_output='{"verdict": "supported"}',
    )

    report = EvaluationReport(
        report_id="report_001",
        dataset_name="test_dataset",
        dataset_version="1.0",
        judge=JudgeType.GROQ,
        model="test-model",
        results=[result],
        metrics={
            "accuracy": 1.0,
        },
        evaluated_count=1,
        skipped_count=0,
    )

    assert report.report_id == "report_001"
    assert report.dataset_name == "test_dataset"
    assert report.dataset_version == "1.0"
    assert report.judge == JudgeType.GROQ
    assert report.model == "test-model"

    assert len(report.results) == 1
    assert report.results[0].example_id == "rag_001"

    assert report.metrics["accuracy"] == 1.0

    assert report.evaluated_count == 1
    assert report.skipped_count == 0

    assert isinstance(report.created_at, datetime)


def test_evaluation_report_defaults_errors_to_empty_list():
    report = EvaluationReport(
        report_id="report_002",
        dataset_name="test_dataset",
        dataset_version="1.0",
        judge=JudgeType.JEV,
        model="jev-test",
        results=[],
        metrics={},
        evaluated_count=0,
        skipped_count=0,
    )

    assert report.errors == []