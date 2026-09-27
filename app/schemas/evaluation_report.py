from datetime import datetime, UTC

from pydantic import BaseModel, Field

from app.schemas.evaluation_result import EvaluationResult, JudgeType


class EvaluationReport(BaseModel):
    """
    Complete evaluation report for a single judge.
    """

    report_id: str
    dataset_name: str
    dataset_version: str
    judge: JudgeType
    model: str

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    results: list[EvaluationResult]
    metrics: dict
    errors: list[dict] = Field(default_factory=list)

    evaluated_count: int
    skipped_count: int