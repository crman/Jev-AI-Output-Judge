from app.clients.groq_client import GroqClient
from app.clients.jev_client import JevClient
from app.evaluation.judge_normalizer import (
    normalize_groq_response,
    normalize_jev_response,
)
from app.schemas.evaluation_result import EvaluationResult, JudgeType
from collections.abc import Callable, Iterable

from app.schemas.evaluation import EvaluationExample
from app.evaluation.prompt_builder import build_groq_prompts
from pathlib import Path

from app.evaluation.dataset_loader import load_dataset

def evaluate_groq_example(
    client: GroqClient,
    example_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> EvaluationResult:
    """Evaluate a single example using Groq and normalize its response."""

    raw_output = client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    return normalize_groq_response(
        example_id=example_id,
        model=model,
        raw_output=raw_output,
    )


def evaluate_jev_example(
    client: JevClient,
    example_id: str,
    state: str | dict | list,
    questions: dict,
) -> EvaluationResult:
    """Evaluate a single example using Jev and normalize its response."""

    raw_output = client.evaluate(
        state=state,
        questions=questions,
    )

    return normalize_jev_response(
        example_id=example_id,
        raw_output=raw_output,
    )
    



def evaluate_example(
    judge: JudgeType,
    client: GroqClient | JevClient,
    example_id: str,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    state: str | dict | list | None = None,
    questions: dict | None = None,
) -> EvaluationResult:
    """
    Evaluate one example using the selected judge.

    Groq requires system_prompt, user_prompt, and model.
    Jev requires state and questions.
    """

    if judge == JudgeType.GROQ:

        if not model or not system_prompt or not user_prompt:
            raise ValueError(
                "Groq evaluation requires model, system_prompt, "
                "and user_prompt."
            )

        return evaluate_groq_example(
            client=client,
            example_id=example_id,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    if judge == JudgeType.JEV:

        if state is None or questions is None:
            raise ValueError(
                "Jev evaluation requires state and questions."
            )

        return evaluate_jev_example(
            client=client,
            example_id=example_id,
            state=state,
            questions=questions,
        )

    raise ValueError(f"Unsupported judge: {judge}")


def evaluate_dataset(
    examples: Iterable[EvaluationExample],
    evaluator: Callable[[EvaluationExample], EvaluationResult],
) -> dict:
    """
    Evaluate a dataset using the supplied single-example evaluator.

    Each example is processed independently. Failures are recorded
    without stopping the remaining batch.
    """

    results: list[EvaluationResult] = []
    errors: list[dict[str, str]] = []

    for example in examples:
        try:
            result = evaluator(example)
            results.append(result)

        except Exception as exc:
            errors.append(
                {
                    "example_id": example.id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    return {
        "total": len(results) + len(errors),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }
    

def evaluate_groq_dataset(
    examples: Iterable[EvaluationExample],
    client: GroqClient,
    model: str,
) -> dict:
    """
    Evaluate a dataset using Groq.

    Builds prompts for each example, evaluates it, and collects
    normalized results while recording individual failures.
    """

    def evaluator(example: EvaluationExample) -> EvaluationResult:
        system_prompt, user_prompt = build_groq_prompts(example)

        return evaluate_groq_example(
            client=client,
            example_id=example.id,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    return evaluate_dataset(
        examples=examples,
        evaluator=evaluator,
    )
    

def evaluate_jev_dataset(
    examples: Iterable[EvaluationExample],
    client: JevClient,
    questions: dict,
) -> dict:
    """
    Evaluate a dataset using Jev.

    Ground-truth labels are intentionally excluded from Jev's state.
    """

    def evaluator(example: EvaluationExample) -> EvaluationResult:
        state = {
            "question": example.question,
            "context": example.context,
            "answer": example.answer,
        }

        return evaluate_jev_example(
            client=client,
            example_id=example.id,
            state=state,
            questions=questions,
        )

    return evaluate_dataset(
        examples=examples,
        evaluator=evaluator,
    )
    

def run_dataset_evaluation(
    dataset_path: str | Path,
    judge: JudgeType,
    client: GroqClient | JevClient,
    *,
    model: str | None = None,
    questions: dict | None = None,
) -> dict:
    """
    Load a dataset and evaluate it using the selected judge.
    """

    dataset = load_dataset(dataset_path)

    if judge == JudgeType.GROQ:
        if not model:
            raise ValueError(
                "A model must be provided for Groq evaluation."
            )

        return evaluate_groq_dataset(
            examples=dataset.examples,
            client=client,
            model=model,
        )

    if judge == JudgeType.JEV:
        if questions is None:
            raise ValueError(
                "Questions must be provided for Jev evaluation."
            )

        return evaluate_jev_dataset(
            examples=dataset.examples,
            client=client,
            questions=questions,
        )

    raise ValueError(f"Unsupported judge: {judge}")