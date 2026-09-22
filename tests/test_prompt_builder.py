import json

from app.evaluation.prompt_builder import build_groq_prompts
from app.schemas.evaluation import EvaluationExample, GroundTruth


def test_build_groq_prompts_includes_example_content():
    example = EvaluationExample(
        id="rag_001",
        question="What is the capital of the UK?",
        context="London is the capital of the UK.",
        answer="London",
        ground_truth=GroundTruth(
            contradiction=False,
            unsupported_claim=False,
        ),
        category="supported_claim",
    )

    system_prompt, user_prompt = build_groq_prompts(example)

    parsed_prompt = json.loads(user_prompt)

    assert "impartial AI response evaluator" in system_prompt
    assert parsed_prompt["question"] == example.question
    assert parsed_prompt["context"] == example.context
    assert parsed_prompt["answer"] == example.answer


def test_build_groq_prompts_excludes_ground_truth():
    example = EvaluationExample(
        id="rag_002",
        question="What is the capital of France?",
        context="Paris is the capital of France.",
        answer="Paris",
        ground_truth=GroundTruth(
            contradiction=False,
            unsupported_claim=False,
        ),
        category="supported_claim",
    )

    _, user_prompt = build_groq_prompts(example)

    assert "ground_truth" not in user_prompt
    assert "contradiction" not in user_prompt
    assert "unsupported_claim" not in user_prompt
