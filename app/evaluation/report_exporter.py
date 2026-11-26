import csv
import json
from pathlib import Path

from app.schemas.comparison_report import ComparisonReport
from app.schemas.evaluation_report import EvaluationReport


def export_report_to_json(
    report: EvaluationReport | ComparisonReport,
    output_path: str | Path,
) -> Path:
    """
    Export an evaluation or comparison report to a JSON file.

    Returns:
        Path to the created report file.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


def export_comparison_to_json(
    report: ComparisonReport,
    output_path: str | Path,
) -> Path:
    """
    Export a comparison report to a JSON file.
    """
    return export_report_to_json(report, output_path)


def export_report_to_csv(
    report: EvaluationReport | ComparisonReport,
    output_path: str | Path,
) -> Path:
    """
    Export evaluation or comparison results to a CSV file.

    Each row represents one evaluation result.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "example_id",
        "judge",
        "model",
        "verdict",
        "confidence",
        "explanation",
        "raw_output",
    ]

    results = []
    if isinstance(report, ComparisonReport):
        for sub_report in report.reports:
            results.extend(sub_report.results)
    else:
        results = report.results

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "example_id": result.example_id,
                    "judge": result.judge.value,
                    "model": result.model,
                    "verdict": result.verdict.value,
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                    "raw_output": result.raw_output,
                }
            )

    return output_path


def export_comparison_to_csv(
    report: ComparisonReport,
    output_path: str | Path,
) -> Path:
    """
    Export a comparison report's evaluation results to a CSV file.
    """
    return export_report_to_csv(report, output_path)