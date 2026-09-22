import json
from unittest.mock import Mock

import pytest

from app.evaluation.judge_normalizer import (
    JudgeNormalizationError,
    normalize_jev_response,
)
from app.evaluation.pipeline import (
    evaluate_dataset,
    evaluate_example,
    evaluate_groq_dataset,
    evaluate_groq_example,
    evaluate_jev_dataset,
    evaluate_jev_example,
)
from app.schemas.evaluation import (
    EvaluationExample,
    GroundTruth,
)
from app.schemas.evaluation_result import (
    EvaluationResult,
    EvaluationVerdict,
    JudgeType,
)
import json

from app.evaluation.pipeline import run_dataset_evaluation

def test_evaluate_groq_example():
    client = Mock()
    client.generate.return_value = (
        '{"verdict": "supported", '
        '"explanation": "The evidence supports the claim."}'
    )

    result = evaluate_groq_example(
        client=client,
        example_id="rag_001",
        model="test-model",
        system_prompt="You are an evaluation judge.",
        user_prompt="Evaluate this claim against the evidence.",
    )

    client.generate.assert_called_once_with(
        system_prompt="You are an evaluation judge.",
        user_prompt="Evaluate this claim against the evidence.",
    )

    assert result.example_id == "rag_001"
    assert result.judge == JudgeType.GROQ
    assert result.model == "test-model"
    assert result.verdict == EvaluationVerdict.SUPPORTED
    assert result.explanation == "The evidence supports the claim."


def test_evaluate_groq_example_preserves_raw_output():
    client = Mock()
    raw_output = '{"verdict": "uncertain", "explanation": "Not enough context."}'
    client.generate.return_value = raw_output

    result = evaluate_groq_example(
        client=client,
        example_id="rag_002",
        model="test-model",
        system_prompt="System prompt",
        user_prompt="User prompt",
    )

    assert result.raw_output == raw_output
    assert result.verdict == EvaluationVerdict.UNCERTAIN
    
def test_evaluate_jev_example():
    client = Mock()
    client.evaluate.return_value = {
        "model": "jev-1.13.0",
        "answers": {
            "hq_supported": {
                "type": "noul",
                "noul": 0.05,
            }
        },
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
        },
    }

    state = {
        "claim": "London is the capital of the UK.",
        "evidence": "London is the capital of the United Kingdom.",
    }
    questions = {
        "hq_supported": {
            "question": "Is the claim supported by the evidence?"
        }
    }

    result = evaluate_jev_example(
        client=client,
        example_id="rag_001",
        state=state,
        questions=questions,
    )

    client.evaluate.assert_called_once_with(
        state=state,
        questions=questions,
    )

    assert result.example_id == "rag_001"
    assert result.judge == JudgeType.JEV
    assert result.model == "jev-1.13.0"
    assert result.verdict == EvaluationVerdict.UNCERTAIN
    assert result.raw_output == client.evaluate.return_value
    assert result.usage == {
        "input_tokens": 100,
        "output_tokens": 20,
    }


def test_evaluate_jev_example_preserves_noul_signal():
    client = Mock()
    client.evaluate.return_value = {
        "model": "jev-1.13.0",
        "answers": {
            "hq_supported": {
                "type": "noul",
                "noul": 0.05,
            }
        },
    }

    result = evaluate_jev_example(
        client=client,
        example_id="rag_002",
        state="Test state",
        questions={"hq_supported": {"question": "Test question"}},
    )

    assert result.raw_output["answers"]["hq_supported"]["noul"] == 0.05
    assert result.verdict == EvaluationVerdict.UNCERTAIN
    assert result.confidence is None
    
def test_unified_pipeline_routes_to_groq():
    client = Mock()
    client.generate.return_value = (
        '{"verdict": "supported", '
        '"explanation": "Evidence supports the claim."}'
    )

    result = evaluate_example(
        judge=JudgeType.GROQ,
        client=client,
        example_id="rag_003",
        model="test-model",
        system_prompt="You are a judge.",
        user_prompt="Evaluate this claim.",
    )

    assert result.judge == JudgeType.GROQ
    assert result.example_id == "rag_003"
    client.generate.assert_called_once()


def test_unified_pipeline_routes_to_jev():
    client = Mock()
    client.evaluate.return_value = {
        "model": "jev-1.13.0",
        "answers": {
            "hq_supported": {
                "type": "noul",
                "noul": 0.05,
            }
        },
    }

    result = evaluate_example(
        judge=JudgeType.JEV,
        client=client,
        example_id="rag_004",
        state="Test state",
        questions={"hq_supported": {"question": "Test?"}},
    )

    assert result.judge == JudgeType.JEV
    assert result.example_id == "rag_004"
    client.evaluate.assert_called_once()


def test_unified_pipeline_requires_groq_prompts():
    with pytest.raises(ValueError, match="Groq evaluation requires"):
        evaluate_example(
            judge=JudgeType.GROQ,
            client=Mock(),
            example_id="rag_005",
        )


def test_unified_pipeline_requires_jev_inputs():
    with pytest.raises(ValueError, match="Jev evaluation requires"):
        evaluate_example(
            judge=JudgeType.JEV,
            client=Mock(),
            example_id="rag_006",
        )
        

def make_example(example_id: str) -> EvaluationExample:
    return EvaluationExample(
        id=example_id,
        question="What is the capital of the UK?",
        context="London is the capital of the UK.",
        answer="London",
        ground_truth=GroundTruth(
            contradiction=False,
            unsupported_claim=False,
        ),
        category="supported_claim",
    )


def test_evaluate_dataset_collects_results():
    examples = [
        make_example("rag_001"),
        make_example("rag_002"),
    ]

    def evaluator(example):
        return EvaluationResult(
            example_id=example.id,
            judge=JudgeType.GROQ,
            model="test-model",
            verdict=EvaluationVerdict.SUPPORTED,
            raw_output='{"verdict": "supported"}',
        )

    report = evaluate_dataset(examples, evaluator)

    assert report["total"] == 2
    assert report["successful"] == 2
    assert report["failed"] == 0
    assert len(report["results"]) == 2
    assert report["errors"] == []


def test_evaluate_dataset_continues_after_failure():
    examples = [
        make_example("rag_001"),
        make_example("rag_002"),
        make_example("rag_003"),
    ]

    def evaluator(example):
        if example.id == "rag_002":
            raise RuntimeError("Simulated judge failure")

        return EvaluationResult(
            example_id=example.id,
            judge=JudgeType.GROQ,
            model="test-model",
            verdict=EvaluationVerdict.SUPPORTED,
            raw_output='{"verdict": "supported"}',
        )

    report = evaluate_dataset(examples, evaluator)

    assert report["total"] == 3
    assert report["successful"] == 2
    assert report["failed"] == 1

    assert [result.example_id for result in report["results"]] == [
        "rag_001",
        "rag_003",
    ]

    assert report["errors"][0]["example_id"] == "rag_002"
    assert report["errors"][0]["error_type"] == "RuntimeError"
    
def test_evaluate_groq_dataset():
    examples = [
        make_example("rag_001"),
        make_example("rag_002"),
    ]

    client = Mock()
    client.generate.side_effect = [
        '{"verdict": "supported", "explanation": "Supported by context."}',
        '{"verdict": "contradicted", "explanation": "Contradicts context."}',
    ]

    report = evaluate_groq_dataset(
        examples=examples,
        client=client,
        model="test-model",
    )

    assert report["total"] == 2
    assert report["successful"] == 2
    assert report["failed"] == 0

    assert report["results"][0].verdict == EvaluationVerdict.SUPPORTED
    assert report["results"][1].verdict == EvaluationVerdict.CONTRADICTED

    assert all(
        result.judge == JudgeType.GROQ
        for result in report["results"]
    )

    assert client.generate.call_count == 2


def test_evaluate_groq_dataset_continues_after_failure():
    examples = [
        make_example("rag_001"),
        make_example("rag_002"),
    ]

    client = Mock()
    client.generate.side_effect = [
        '{"verdict": "supported", "explanation": "Supported."}',
        RuntimeError("Simulated Groq failure"),
    ]

    report = evaluate_groq_dataset(
        examples=examples,
        client=client,
        model="test-model",
    )

    assert report["total"] == 2
    assert report["successful"] == 1
    assert report["failed"] == 1

    assert report["results"][0].example_id == "rag_001"
    assert report["errors"][0]["example_id"] == "rag_002"
    assert report["errors"][0]["error_type"] == "RuntimeError"
    
def test_evaluate_jev_dataset():
    examples = [
        make_example("rag_001"),
        make_example("rag_002"),
    ]

    questions = {
        "hq_supported": {
            "question": "Is the answer supported by the context?"
        }
    }

    client = Mock()
    client.evaluate.side_effect = [
        {
            "model": "jev-1.13.0",
            "answers": {
                "hq_supported": {
                    "type": "noul",
                    "noul": 0.05,
                }
            },
            "usage": {"input_tokens": 100, "output_tokens": 20},
        },
        {
            "model": "jev-1.13.0",
            "answers": {
                "hq_supported": {
                    "type": "noul",
                    "noul": 0.15,
                }
            },
        },
    ]

    report = evaluate_jev_dataset(
        examples=examples,
        client=client,
        questions=questions,
    )

    assert report["total"] == 2
    assert report["successful"] == 2
    assert report["failed"] == 0
    assert len(report["results"]) == 2

    assert all(
        result.judge == JudgeType.JEV
        for result in report["results"]
    )
    assert all(
        result.verdict == EvaluationVerdict.UNCERTAIN
        for result in report["results"]
    )

    assert client.evaluate.call_count == 2

    first_call = client.evaluate.call_args_list[0]
    assert first_call.kwargs["state"] == {
        "question": examples[0].question,
        "context": examples[0].context,
        "answer": examples[0].answer,
    }
    assert first_call.kwargs["questions"] == questions

    # Ground truth must not be sent to Jev.
    assert "ground_truth" not in first_call.kwargs["state"]


def test_evaluate_jev_dataset_continues_after_failure():
    examples = [
        make_example("rag_001"),
        make_example("rag_002"),
    ]

    client = Mock()
    client.evaluate.side_effect = [
        {
            "model": "jev-1.13.0",
            "answers": {
                "hq_supported": {
                    "type": "noul",
                    "noul": 0.05,
                }
            },
        },
        RuntimeError("Simulated Jev failure"),
    ]

    report = evaluate_jev_dataset(
        examples=examples,
        client=client,
        questions={
            "hq_supported": {
                "question": "Is the answer supported?"
            }
        },
    )

    assert report["total"] == 2
    assert report["successful"] == 1
    assert report["failed"] == 1

    assert report["results"][0].example_id == "rag_001"
    assert report["errors"][0]["example_id"] == "rag_002"
    assert report["errors"][0]["error_type"] == "RuntimeError"
    
def write_test_dataset(tmp_path):
    """Create a valid one-example evaluation dataset."""
    dataset_path = tmp_path / "test_dataset.json"

    dataset = {
        "dataset_name": "test_dataset",
        "version": "1.0",
        "examples": [
            {
                "id": "test_001",
                "question": "What is the capital of France?",
                "context": "Paris is the capital of France.",
                "answer": "Paris is the capital of France.",
                "ground_truth": {
                    "contradiction": False,
                    "unsupported_claim": False,
                },
                "category": "factual",
            }
        ],
    }

    dataset_path.write_text(
        json.dumps(dataset),
        encoding="utf-8",
    )

    return dataset_path


def test_run_dataset_evaluation_routes_to_groq(
    tmp_path,
    monkeypatch,
):
    """Verify dataset evaluation routes to the Groq adapter."""
    dataset_path = write_test_dataset(tmp_path)
    test_client = object()

    expected_report = {
        "results": [],
        "errors": [],
    }

    def mock_groq_dataset(*, examples, client, model):
        assert len(examples) == 1
        assert examples[0].id == "test_001"
        assert client is test_client
        assert model == "test-model"

        return expected_report

    monkeypatch.setattr(
        "app.evaluation.pipeline.evaluate_groq_dataset",
        mock_groq_dataset,
    )

    report = run_dataset_evaluation(
        dataset_path,
        JudgeType.GROQ,
        test_client,
        model="test-model",
    )

    assert report == expected_report

def test_run_dataset_evaluation_routes_to_jev(
    tmp_path,
    monkeypatch,
):
    """Verify dataset evaluation routes to the Jev adapter."""
    dataset_path = write_test_dataset(tmp_path)
    test_client = object()

    questions = ["Is the answer supported by the context?"]

    expected_report = {
        "results": [],
        "errors": [],
    }

    def mock_jev_dataset(*, examples, client, questions):
        assert len(examples) == 1
        assert examples[0].id == "test_001"
        assert client is test_client
        assert questions == ["Is the answer supported by the context?"]

        return expected_report

    monkeypatch.setattr(
        "app.evaluation.pipeline.evaluate_jev_dataset",
        mock_jev_dataset,
    )

    report = run_dataset_evaluation(
        dataset_path,
        JudgeType.JEV,
        test_client,
        questions=questions,
    )

    assert report == expected_report