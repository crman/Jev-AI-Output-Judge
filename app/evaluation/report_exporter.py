import csv
import json
from pathlib import Path

from app.schemas.evaluation_report import EvaluationReport


def export_report_to_json(
    report: EvaluationReport,
    output_path: str | Path,
) -> Path:
    """
    Export an evaluation report to a JSON file.

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


def export_report_to_csv(
    report: EvaluationReport,
    output_path: str | Path,
) -> Path:
    """
    Export evaluation results to a CSV file.

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

        for result in report.results:
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