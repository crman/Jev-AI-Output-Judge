from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.evaluation import GroundTruth
from app.schemas.evaluation_result import EvaluationResult, EvaluationVerdict, JudgeType


@dataclass
class BinaryClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

@dataclass
class EvaluationMetrics:
    contradiction: BinaryClassificationMetrics
    unsupported_claim: BinaryClassificationMetrics
    evaluated_count: int
    skipped_count: int


@dataclass
class JudgeEvaluationMetrics:
    judge: JudgeType
    model: str
    metrics: EvaluationMetrics
    

@dataclass
class JudgeComparisonMetrics:
    judges: list[JudgeEvaluationMetrics]

def calculate_binary_metrics(
    actual: list[bool],
    predicted: list[bool],
) -> BinaryClassificationMetrics:
    """
    Calculate binary classification metrics.

    Args:
        actual: Ground-truth boolean labels.
        predicted: Predicted boolean labels.

    Returns:
        BinaryClassificationMetrics containing classification metrics.

    Raises:
        ValueError: If actual and predicted have different lengths
            or are empty.
    """
    if not actual or not predicted:
        raise ValueError("actual and predicted must not be empty.")

    if len(actual) != len(predicted):
        raise ValueError(
            "actual and predicted must have the same length."
        )

    true_positive = sum(
        actual_value and predicted_value
        for actual_value, predicted_value in zip(actual, predicted)
    )

    true_negative = sum(
        not actual_value and not predicted_value
        for actual_value, predicted_value in zip(actual, predicted)
    )

    false_positive = sum(
        not actual_value and predicted_value
        for actual_value, predicted_value in zip(actual, predicted)
    )

    false_negative = sum(
        actual_value and not predicted_value
        for actual_value, predicted_value in zip(actual, predicted)
    )

    total = len(actual)

    accuracy = (true_positive + true_negative) / total

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    
    


def verdict_to_labels(
    verdict: EvaluationVerdict,
) -> tuple[bool | None, bool | None]:
    """
    Convert an evaluation verdict into contradiction and unsupported-claim labels.

    Returns:
        A tuple of:
        (contradiction, unsupported_claim)

        None indicates that the verdict cannot be mapped confidently.
    """
    if verdict == EvaluationVerdict.SUPPORTED:
        return False, False

    if verdict == EvaluationVerdict.CONTRADICTED:
        return True, False

    if verdict == EvaluationVerdict.INSUFFICIENT_EVIDENCE:
        return False, True

    if verdict == EvaluationVerdict.UNCERTAIN:
        return None, None

    raise ValueError(f"Unsupported evaluation verdict: {verdict}")



def calculate_evaluation_metrics(
    results: Sequence[EvaluationResult],
    ground_truth: dict[str, GroundTruth],
) -> EvaluationMetrics:
    """
    Calculate evaluation metrics for contradiction and unsupported-claim labels.

    Results with an UNCERTAIN verdict are excluded from metric calculation.
    """

    contradiction_actual: list[bool] = []
    contradiction_predicted: list[bool] = []

    unsupported_actual: list[bool] = []
    unsupported_predicted: list[bool] = []

    skipped_count = 0

    for result in results:
        truth = ground_truth.get(result.example_id)

        if truth is None:
            raise ValueError(
                f"Missing ground truth for example: {result.example_id}"
            )

        contradiction, unsupported_claim = verdict_to_labels(
            result.verdict
        )

        if contradiction is None or unsupported_claim is None:
            skipped_count += 1
            continue

        contradiction_actual.append(truth.contradiction)
        contradiction_predicted.append(contradiction)

        unsupported_actual.append(truth.unsupported_claim)
        unsupported_predicted.append(unsupported_claim)

    if not contradiction_actual:
        raise ValueError(
            "No evaluable results available for metric calculation."
        )

    return EvaluationMetrics(
        contradiction=calculate_binary_metrics(
            contradiction_actual,
            contradiction_predicted,
        ),
        unsupported_claim=calculate_binary_metrics(
            unsupported_actual,
            unsupported_predicted,
        ),
        evaluated_count=len(contradiction_actual),
        skipped_count=skipped_count,
    )
    

def calculate_judge_metrics(
    results: Sequence[EvaluationResult],
    ground_truth: dict[str, GroundTruth],
) -> JudgeEvaluationMetrics:
    """
    Calculate evaluation metrics for a single judge.
    """
    if not results:
        raise ValueError("results must not be empty.")

    metrics = calculate_evaluation_metrics(
        results,
        ground_truth,
    )

    return JudgeEvaluationMetrics(
        judge=results[0].judge,
        model=results[0].model,
        metrics=metrics,
    )
    
    
def calculate_judge_comparison(
    results_by_judge: dict[JudgeType, Sequence[EvaluationResult]],
    ground_truth: dict[str, GroundTruth],
) -> JudgeComparisonMetrics:
    """
    Calculate metrics for multiple judges.

    Each judge is evaluated independently against the same ground truth.
    """
    if not results_by_judge:
        raise ValueError("results_by_judge must not be empty.")

    judge_metrics = []

    for judge, results in results_by_judge.items():
        if not results:
            raise ValueError(
                f"No results provided for judge: {judge.value}"
            )

        metrics = calculate_judge_metrics(
            results,
            ground_truth,
        )

        judge_metrics.append(metrics)

    return JudgeComparisonMetrics(
        judges=judge_metrics,
    )