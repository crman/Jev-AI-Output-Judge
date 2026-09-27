from uuid import uuid4

from app.schemas.comparison_report import ComparisonReport
from app.schemas.evaluation_report import EvaluationReport


def build_comparison_report(
    reports: list[EvaluationReport],
) -> ComparisonReport:
    """
    Build a side-by-side comparison report.

    All reports must belong to the same dataset and dataset version.
    """

    if not reports:
        raise ValueError("At least one evaluation report is required.")

    dataset_name = reports[0].dataset_name
    dataset_version = reports[0].dataset_version

    for report in reports:
        if report.dataset_name != dataset_name:
            raise ValueError(
                "All reports must use the same dataset."
            )

        if report.dataset_version != dataset_version:
            raise ValueError(
                "All reports must use the same dataset version."
            )

    judges = [report.judge for report in reports]

    if len(judges) != len(set(judges)):
        raise ValueError(
            "Comparison cannot contain duplicate judges."
        )

    return ComparisonReport(
        comparison_id=str(uuid4()),
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        reports=reports,
    )