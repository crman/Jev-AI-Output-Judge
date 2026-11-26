import argparse

from app.clients.groq_client import GroqClient
from app.clients.jev_client import JevClient
from app.config import settings
from app.evaluation.comparison_builder import build_comparison_report
from app.evaluation.dataset_loader import load_dataset
from app.evaluation.jev_questions import JEV_EVALUATION_QUESTIONS
from app.evaluation.pipeline import run_dataset_evaluation
from app.evaluation.report_builder import build_evaluation_report
from app.evaluation.report_exporter import (
    export_report_to_csv,
    export_report_to_json,
)
from app.schemas.comparison_report import ComparisonReport
from app.schemas.evaluation import EvaluationDataset
from app.schemas.evaluation_report import EvaluationReport
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
        choices=["jev", "groq", "both"],
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


def _run_groq_evaluation(dataset_path: str) -> dict:
    client = GroqClient()
    try:
        return run_dataset_evaluation(
            dataset_path,
            JudgeType.GROQ,
            client,
            model=settings.groq_model,
        )
    finally:
        client.close()


def _run_jev_evaluation(dataset_path: str) -> dict:
    client = JevClient()
    try:
        return run_dataset_evaluation(
            dataset_path,
            JudgeType.JEV,
            client,
            questions=JEV_EVALUATION_QUESTIONS,
            support_threshold=settings.jev_support_threshold,
        )
    finally:
        client.close()


def _print_judge_summary(judge: JudgeType, evaluation_data: dict) -> None:
    results = evaluation_data["results"]
    errors = evaluation_data["errors"]

    print(f"Judge: {judge.value}")
    print(f"Evaluated: {len(results)}")
    print(f"Errors: {len(errors)}")

    if judge == JudgeType.JEV and results:
        print("\nJev signals:")

        for result in results:
            noul = result.raw_output["answers"]["hq_supported"]["noul"]
            print(
                f"{result.example_id}: "
                f"noul={noul:.4f}, "
                f"verdict={result.verdict.value}"
            )


def _build_judge_report(
    dataset: EvaluationDataset,
    judge: JudgeType,
    evaluation_data: dict,
) -> EvaluationReport | None:
    results = evaluation_data["results"]
    errors = evaluation_data["errors"]

    if not results:
        return None

    model = results[0].model
    return build_evaluation_report(
        dataset=dataset,
        judge=judge,
        model=model,
        results=results,
        errors=errors,
    )


def run_cli(
    args: list[str] | None = None,
) -> EvaluationReport | ComparisonReport | None:
    parsed_args = parse_args(args)
    dataset = load_dataset(parsed_args.dataset)

    if parsed_args.judge == "both":
        jev_data = _run_jev_evaluation(parsed_args.dataset)
        _print_judge_summary(JudgeType.JEV, jev_data)

        print()

        groq_data = _run_groq_evaluation(parsed_args.dataset)
        _print_judge_summary(JudgeType.GROQ, groq_data)

        jev_report = _build_judge_report(dataset, JudgeType.JEV, jev_data)
        groq_report = _build_judge_report(dataset, JudgeType.GROQ, groq_data)

        if jev_report is None or groq_report is None:
            raise ValueError(
                "Cannot build comparison report: evaluation failed for one or more judges."
            )

        comparison_report = build_comparison_report([jev_report, groq_report])

        if parsed_args.output:
            if parsed_args.format == "csv":
                export_report_to_csv(comparison_report, parsed_args.output)
            else:
                export_report_to_json(comparison_report, parsed_args.output)

            print(f"\nReport written to: {parsed_args.output}")

        return comparison_report

    judge = JudgeType(parsed_args.judge)

    if judge == JudgeType.GROQ:
        evaluation_data = _run_groq_evaluation(parsed_args.dataset)
    else:
        evaluation_data = _run_jev_evaluation(parsed_args.dataset)

    _print_judge_summary(judge, evaluation_data)

    report = _build_judge_report(dataset, judge, evaluation_data)

    if parsed_args.output:
        if report is None:
            raise ValueError(
                "Cannot export report: no evaluation results were produced."
            )

        if parsed_args.format == "csv":
            export_report_to_csv(report, parsed_args.output)
        else:
            export_report_to_json(report, parsed_args.output)

        print(f"\nReport written to: {parsed_args.output}")

    return report


if __name__ == "__main__":
    run_cli()
    run_cli()