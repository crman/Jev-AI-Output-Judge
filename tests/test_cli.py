import pytest

from app.cli import create_parser, parse_args


def test_parser_requires_dataset():
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--judge", "groq"])


def test_parser_requires_judge():
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--dataset", "data/test.json"]
        )


def test_parse_groq_arguments():
    args = parse_args(
        [
            "--dataset",
            "data/evaluation_dataset.json",
            "--judge",
            "groq",
        ]
    )

    assert args.dataset == "data/evaluation_dataset.json"
    assert args.judge == "groq"
    assert args.output is None
    assert args.format == "json"


def test_parse_jev_arguments():
    args = parse_args(
        [
            "--dataset",
            "data/evaluation_dataset.json",
            "--judge",
            "jev",
            "--output",
            "results/jev_report.csv",
            "--format",
            "csv",
        ]
    )

    assert args.dataset == "data/evaluation_dataset.json"
    assert args.judge == "jev"
    assert args.output == "results/jev_report.csv"
    assert args.format == "csv"


def test_invalid_judge_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset",
                "data/evaluation_dataset.json",
                "--judge",
                "invalid",
            ]
        )


def test_invalid_format_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset",
                "data/evaluation_dataset.json",
                "--judge",
                "groq",
                "--format",
                "xml",
            ]
        )