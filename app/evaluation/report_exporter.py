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