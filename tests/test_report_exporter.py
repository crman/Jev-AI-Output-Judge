import csv
import json

from app.evaluation.report_exporter import (
    export_report_to_csv,
    export_report_to_json,
)
from app.schemas.evaluation_report import EvaluationReport
from app.schemas.evaluation_result import (
    EvaluationResult,
    EvaluationVerdict,
    JudgeType,
)


def make_report():
    result = EvaluationResult(
        example_id="rag_001",
        judge=JudgeType.GROQ,
        model="test-model",
        verdict=EvaluationVerdict.SUPPORTED,
        raw_output='{"verdict": "supported"}',
    )

    return EvaluationReport(
        report_id="report_001",
        dataset_name="test_dataset",
        dataset_version="1.0",
        judge=JudgeType.GROQ,
        model="test-model",
        results=[result],
        metrics={
            "contradiction": {
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
        },
        evaluated_count=1,
        skipped_count=0,
    )


def test_export_report_to_json(tmp_path):
    report = make_report()

    output_path = tmp_path / "report.json"

    returned_path = export_report_to_json(
        report,
        output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()

    data = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert data["report_id"] == "report_001"
    assert data["dataset_name"] == "test_dataset"
    assert data["dataset_version"] == "1.0"
    assert data["judge"] == "groq"
    assert data["model"] == "test-model"

    assert len(data["results"]) == 1
    assert data["results"][0]["example_id"] == "rag_001"

    assert data["metrics"]["contradiction"]["accuracy"] == 1.0


def test_export_report_creates_parent_directory(tmp_path):
    report = make_report()

    output_path = (
        tmp_path
        / "nested"
        / "reports"
        / "report.json"
    )

    export_report_to_json(
        report,
        output_path,
    )

    assert output_path.exists()
    
def test_export_report_to_csv(tmp_path):
    report = make_report()

    output_path = tmp_path / "report.csv"

    returned_path = export_report_to_csv(
        report,
        output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1

    row = rows[0]

    assert row["example_id"] == "rag_001"
    assert row["judge"] == "groq"
    assert row["model"] == "test-model"
    assert row["verdict"] == "supported"
    assert row["raw_output"] == '{"verdict": "supported"}'
    
def test_export_report_to_csv_creates_parent_directory(tmp_path):
    report = make_report()

    output_path = (
        tmp_path
        / "nested"
        / "reports"
        / "report.csv"
    )

    export_report_to_csv(
        report,
        output_path,
    )

    assert output_path.exists()