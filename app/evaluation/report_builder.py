
from uuid import uuid4

from app.evaluation.metrics import calculate_judge_metrics
from app.schemas.evaluation import EvaluationDataset
from app.schemas.evaluation_report import EvaluationReport
from app.schemas.evaluation_result import EvaluationResult, JudgeType


def build_evaluation_report(
    dataset: EvaluationDataset,
    judge: JudgeType,
    model: str,
    results: list[EvaluationResult],
    errors: list[dict] | None = None,
) -> EvaluationReport:
    """Build a report for one judge using dataset ground truth."""

    if not results:
        raise ValueError("At least one evaluation result is required.")

    if any(
        result.judge != judge or result.model != model
        for result in results
    ):
        raise ValueError(
            "All results must match the selected judge and model."
        )

    example_ids = [result.example_id for result in results]

    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Duplicate example IDs in evaluation results.")

    ground_truth = {
        example.id: example.ground_truth
        for example in dataset.examples
    }

    judge_metrics = calculate_judge_metrics(
        results,
        ground_truth,
    )

    metrics = judge_metrics.metrics

    return EvaluationReport(
        report_id=str(uuid4()),
        dataset_name=dataset.dataset_name,
        dataset_version=dataset.version,
        judge=judge,
        model=model,
        results=results,
        metrics={
            "contradiction": vars(metrics.contradiction),
            "unsupported_claim": vars(metrics.unsupported_claim),
        },
        errors=errors or [],
        evaluated_count=metrics.evaluated_count,
        skipped_count=metrics.skipped_count,
    )