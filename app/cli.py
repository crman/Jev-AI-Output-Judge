import argparse

from app.clients.groq_client import GroqClient
from app.clients.jev_client import JevClient
from app.config import settings
from app.evaluation.jev_questions import JEV_EVALUATION_QUESTIONS
from app.evaluation.pipeline import run_dataset_evaluation
from app.schemas.evaluation_result import JudgeType


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LLM responses using Jev or a Groq-hosted LLM judge."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the evaluation dataset JSON file.",
    )

    parser.add_argument(
        "--judge",
        required=True,
        choices=["jev", "groq"],
        help="Judge to use for evaluation.",
    )

    parser.add_argument(
        "--output",
        help="Optional path for the evaluation report.",
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output report format. Defaults to JSON.",
    )

    return parser


def parse_args(
    args: list[str] | None = None,
) -> argparse.Namespace:
    parser = create_parser()
    return parser.parse_args(args)


def run_cli(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args)

    judge = JudgeType(parsed_args.judge)

    if judge == JudgeType.GROQ:
        client = GroqClient()

        try:
            report = run_dataset_evaluation(
                parsed_args.dataset,
                judge,
                client,
                model=settings.groq_model,
            )
        finally:
            client.close()

    else:
        client = JevClient()

        try:
            report = run_dataset_evaluation(
                parsed_args.dataset,
                judge,
                client,
                questions=JEV_EVALUATION_QUESTIONS,
                support_threshold=settings.jev_support_threshold,
            )
        finally:
            client.close()

    print(f"Judge: {judge.value}")
    print(f"Evaluated: {len(report['results'])}")
    print(f"Errors: {len(report['errors'])}")

    if judge == JudgeType.JEV:
        print("\nJev signals:")

        for result in report["results"]:
            noul = result.raw_output["answers"]["hq_supported"]["noul"]
            print(
                f"{result.example_id}: "
                f"noul={noul:.4f}, "
                f"verdict={result.verdict.value}"
            )
        
    
if __name__ == "__main__":
    run_cli()