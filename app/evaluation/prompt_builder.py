import json

from app.schemas.evaluation import EvaluationExample


def build_groq_prompts(
    example: EvaluationExample,
) -> tuple[str, str]:
    """
    Build system and user prompts for Groq evaluation.

    Ground-truth labels are intentionally excluded from the prompt.
    """

    system_prompt = """
You are an impartial AI response evaluator.

Evaluate the answer using only the provided question and context.

Choose exactly one verdict:
- supported: The context supports the answer.
- contradicted: The context contradicts the answer.
- insufficient_evidence: The context does not provide enough information.
- uncertain: The evidence is ambiguous or the verdict cannot be determined.

Return only a valid JSON object with these fields:
{
  "verdict": "supported | contradicted | insufficient_evidence | uncertain",
  "explanation": "Brief explanation based on the context"
}

Do not include Markdown fences or additional text.
""".strip()

    user_prompt = json.dumps(
        {
            "question": example.question,
            "context": example.context,
            "answer": example.answer,
        },
        ensure_ascii=False,
        indent=2,
    )

    return system_prompt, user_prompt