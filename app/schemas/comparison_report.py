from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.schemas.evaluation_report import EvaluationReport


class ComparisonReport(BaseModel):
    """
    Side-by-side comparison of multiple evaluation reports.
    """

    comparison_id: str
    dataset_name: str
    dataset_version: str

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    reports: list[EvaluationReport]