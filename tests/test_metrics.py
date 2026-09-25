import pytest

from app.evaluation.metrics import (
    BinaryClassificationMetrics,
    EvaluationMetrics,
    JudgeComparisonMetrics,
    JudgeEvaluationMetrics,
    calculate_binary_metrics,
    calculate_evaluation_metrics,
    calculate_judge_comparison,
    calculate_judge_metrics,
    verdict_to_labels,
)
from app.schemas.evaluation import GroundTruth
from app.schemas.evaluation_result import (
    EvaluationResult,
    EvaluationVerdict,
    JudgeType,
)


def test_calculate_binary_metrics():
    actual = [True, True, False, False]
    predicted = [True, False, False, True]

    result = calculate_binary_metrics(actual, predicted)

    assert isinstance(result, BinaryClassificationMetrics)

    assert result.true_positive == 1
    assert result.true_negative == 1
    assert result.false_positive == 1
    assert result.false_negative == 1

    assert result.accuracy == 0.5
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1 == 0.5


def test_perfect_predictions():
    actual = [True, False, True, False]
    predicted = [True, False, True, False]

    result = calculate_binary_metrics(actual, predicted)

    assert result.accuracy == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_all_predictions_are_negative():
    actual = [False, False, False]
    predicted = [False, False, False]

    result = calculate_binary_metrics(actual, predicted)

    assert result.accuracy == 1.0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_empty_input_raises_error():
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_binary_metrics([], [])


def test_different_lengths_raise_error():
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        calculate_binary_metrics(
            [True, False],
            [True],
        )
    
def test_supported_verdict_maps_to_supported_labels():
    contradiction, unsupported_claim = verdict_to_labels(
        EvaluationVerdict.SUPPORTED
    )

    assert contradiction is False
    assert unsupported_claim is False


def test_contradicted_verdict_maps_to_contradiction():
    contradiction, unsupported_claim = verdict_to_labels(
        EvaluationVerdict.CONTRADICTED
    )

    assert contradiction is True
    assert unsupported_claim is False


def test_insufficient_evidence_maps_to_unsupported_claim():
    contradiction, unsupported_claim = verdict_to_labels(
        EvaluationVerdict.INSUFFICIENT_EVIDENCE
    )

    assert contradiction is False
    assert unsupported_claim is True


def test_uncertain_verdict_has_no_labels():
    contradiction, unsupported_claim = verdict_to_labels(
        EvaluationVerdict.UNCERTAIN
    )

    assert contradiction is None
    assert unsupported_claim is None
    
def make_result(
    example_id: str,
    verdict: EvaluationVerdict,
) -> EvaluationResult:
    return EvaluationResult(
        example_id=example_id,
        judge=JudgeType.GROQ,
        model="test-model",
        verdict=verdict,
        raw_output="{}",
    )


def test_calculate_evaluation_metrics():
    results = [
        make_result("example_1", EvaluationVerdict.CONTRADICTED),
        make_result(
            "example_2",
            EvaluationVerdict.SUPPORTED,
        ),
        make_result(
            "example_3",
            EvaluationVerdict.INSUFFICIENT_EVIDENCE,
        ),
        make_result(
            "example_4",
            EvaluationVerdict.SUPPORTED,
        ),
    ]

    ground_truth = {
        "example_1": GroundTruth(
            contradiction=True,
            unsupported_claim=False,
        ),
        "example_2": GroundTruth(
            contradiction=False,
            unsupported_claim=False,
        ),
        "example_3": GroundTruth(
            contradiction=False,
            unsupported_claim=True,
        ),
        "example_4": GroundTruth(
            contradiction=False,
            unsupported_claim=False,
        ),
    }

    metrics = calculate_evaluation_metrics(
        results,
        ground_truth,
    )

    assert isinstance(metrics, EvaluationMetrics)

    assert metrics.evaluated_count == 4
    assert metrics.skipped_count == 0

    assert metrics.contradiction.accuracy == 1.0
    assert metrics.contradiction.precision == 1.0
    assert metrics.contradiction.recall == 1.0
    assert metrics.contradiction.f1 == 1.0

    assert metrics.unsupported_claim.accuracy == 1.0
    assert metrics.unsupported_claim.precision == 1.0
    assert metrics.unsupported_claim.recall == 1.0
    assert metrics.unsupported_claim.f1 == 1.0
    
def test_uncertain_results_are_skipped():
    results = [
        make_result("example_1", EvaluationVerdict.SUPPORTED),
        make_result("example_2", EvaluationVerdict.UNCERTAIN),
    ]

    ground_truth = {
        "example_1": GroundTruth(
            contradiction=False,
            unsupported_claim=False,
        ),
        "example_2": GroundTruth(
            contradiction=True,
            unsupported_claim=False,
        ),
    }

    metrics = calculate_evaluation_metrics(
        results,
        ground_truth,
    )

    assert metrics.evaluated_count == 1
    assert metrics.skipped_count == 1
    
def test_missing_ground_truth_raises_error():
    results = [
        make_result(
            "missing_example",
            EvaluationVerdict.SUPPORTED,
        )
    ]

    with pytest.raises(
        ValueError,
        match="Missing ground truth",
    ):
        calculate_evaluation_metrics(
            results,
            {},
        )
        
def test_only_uncertain_results_raise_error():
    results = [
        make_result(
            "example_1",
            EvaluationVerdict.UNCERTAIN,
        )
    ]

    ground_truth = {
        "example_1": GroundTruth(
            contradiction=False,
            unsupported_claim=False,
        )
    }

    with pytest.raises(
        ValueError,
        match="No evaluable results",
    ):
        calculate_evaluation_metrics(
            results,
            ground_truth,
        )
        
def test_calculate_judge_metrics():
    results = [
        make_result(
            "example_1",
            EvaluationVerdict.CONTRADICTED,
        ),
        make_result(
            "example_2",
            EvaluationVerdict.SUPPORTED,
        ),
    ]

    ground_truth = {
        "example_1": GroundTruth(
            contradiction=True,
            unsupported_claim=False,
        ),
        "example_2": GroundTruth(
            contradiction=False,
            unsupported_claim=False,
        ),
    }

    result = calculate_judge_metrics(
        results,
        ground_truth,
    )

    assert isinstance(result, JudgeEvaluationMetrics)
    assert result.judge == JudgeType.GROQ
    assert result.model == "test-model"

    assert result.metrics.evaluated_count == 2
    assert result.metrics.skipped_count == 0
    assert result.metrics.contradiction.accuracy == 1.0
    
def test_calculate_judge_metrics_requires_results():
    with pytest.raises(
        ValueError,
        match="results must not be empty",
    ):
        calculate_judge_metrics(
            [],
            {},
        )
        
def test_calculate_judge_comparison():
    groq_results = [
        make_result(
            "example_1",
            EvaluationVerdict.CONTRADICTED,
        ),
        make_result(
            "example_2",
            EvaluationVerdict.SUPPORTED,
        ),
    ]

    jev_results = [
        EvaluationResult(
            example_id="example_1",
            judge=JudgeType.JEV,
            model="jev-test",
            verdict=EvaluationVerdict.CONTRADICTED,
            raw_output="jev output",
        ),
        EvaluationResult(
            example_id="example_2",
            judge=JudgeType.JEV,
            model="jev-test",
            verdict=EvaluationVerdict.SUPPORTED,
            raw_output="jev output",
        ),
    ]

    ground_truth = {
        "example_1": GroundTruth(
            contradiction=True,
            unsupported_claim=False,
        ),
        "example_2": GroundTruth(
            contradiction=False,
            unsupported_claim=False,
        ),
    }

    result = calculate_judge_comparison(
        {
            JudgeType.GROQ: groq_results,
            JudgeType.JEV: jev_results,
        },
        ground_truth,
    )

    assert isinstance(result, JudgeComparisonMetrics)

    assert len(result.judges) == 2

    assert result.judges[0].judge == JudgeType.GROQ
    assert result.judges[0].model == "test-model"

    assert result.judges[1].judge == JudgeType.JEV
    assert result.judges[1].model == "jev-test"

    assert result.judges[0].metrics.contradiction.accuracy == 1.0
    assert result.judges[1].metrics.contradiction.accuracy == 1.0
    
def test_calculate_judge_comparison_requires_judges():
    with pytest.raises(
        ValueError,
        match="results_by_judge must not be empty",
    ):
        calculate_judge_comparison(
            {},
            {},
        )