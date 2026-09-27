import argparse


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