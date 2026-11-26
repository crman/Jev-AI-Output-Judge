import json
import pytest

from app.cli import create_parser, parse_args
from app.schemas.comparison_report import ComparisonReport
from app.schemas.evaluation_result import (
    EvaluationResult,
    EvaluationVerdict,
    JudgeType,
)


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


def test_parse_both_arguments():
    args = parse_args(
        [
            "--dataset",
            "data/evaluation_dataset.json",
            "--judge",
            "both",
            "--output",
            "results/comparison.json",
        ]
    )

    assert args.dataset == "data/evaluation_dataset.json"
    assert args.judge == "both"
    assert args.output == "results/comparison.json"
    assert args.format == "json"



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


def _make_mock_result(
    example_id: str = "rag_001",
    judge: JudgeType = JudgeType.GROQ,
    model: str = "test-model",
    verdict: EvaluationVerdict = EvaluationVerdict.SUPPORTED,
    raw_output: dict | str = '{"verdict": "supported"}',
) -> EvaluationResult:
    return EvaluationResult(
        example_id=example_id,
        judge=judge,
        model=model,
        verdict=verdict,
        raw_output=raw_output,
    )


def test_run_cli_exports_json_report(monkeypatch, tmp_path, capsys):
    from app import cli
    from app.schemas.evaluation_result import EvaluationResult, EvaluationVerdict, JudgeType

    mock_result = _make_mock_result(
        example_id="rag_001",
        judge=JudgeType.GROQ,
        model="groq-test-model",
        verdict=EvaluationVerdict.SUPPORTED,
    )

    def mock_run_dataset_evaluation(*args, **kwargs):
        return {
            "total": 1,
            "successful": 1,
            "failed": 0,
            "results": [mock_result],
            "errors": [],
        }

    monkeypatch.setattr(
        cli, "run_dataset_evaluation", mock_run_dataset_evaluation
    )

    output_file = tmp_path / "report.json"
    cli.run_cli(
        [
            "--dataset",
            "data/evaluation_dataset.json",
            "--judge",
            "groq",
            "--output",
            str(output_file),
            "--format",
            "json",
        ]
    )

    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert data["judge"] == "groq"
    assert data["model"] == "groq-test-model"
    assert len(data["results"]) == 1

    captured = capsys.readouterr().out
    assert "Judge: groq" in captured
    assert "Evaluated: 1" in captured
    assert f"Report written to: {output_file}" in captured


def test_run_cli_exports_csv_report(monkeypatch, tmp_path, capsys):
    import csv
    from app import cli
    from app.schemas.evaluation_result import EvaluationVerdict, JudgeType

    mock_result = _make_mock_result(
        example_id="rag_001",
        judge=JudgeType.GROQ,
        model="groq-test-model",
        verdict=EvaluationVerdict.SUPPORTED,
    )

    def mock_run_dataset_evaluation(*args, **kwargs):
        return {
            "total": 1,
            "successful": 1,
            "failed": 0,
            "results": [mock_result],
            "errors": [],
        }

    monkeypatch.setattr(
        cli, "run_dataset_evaluation", mock_run_dataset_evaluation
    )

    output_file = tmp_path / "report.csv"
    cli.run_cli(
        [
            "--dataset",
            "data/evaluation_dataset.json",
            "--judge",
            "groq",
            "--output",
            str(output_file),
            "--format",
            "csv",
        ]
    )

    assert output_file.exists()
    with output_file.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["example_id"] == "rag_001"
    assert rows[0]["verdict"] == "supported"

    captured = capsys.readouterr().out
    assert f"Report written to: {output_file}" in captured


def test_run_cli_without_output(monkeypatch, capsys):
    from app import cli
    from app.schemas.evaluation_result import EvaluationVerdict, JudgeType

    mock_result = _make_mock_result(
        example_id="rag_001",
        judge=JudgeType.GROQ,
        model="groq-test-model",
        verdict=EvaluationVerdict.SUPPORTED,
    )

    monkeypatch.setattr(
        cli,
        "run_dataset_evaluation",
        lambda *args, **kwargs: {
            "total": 1,
            "successful": 1,
            "failed": 0,
            "results": [mock_result],
            "errors": [],
        },
    )

    report = cli.run_cli(
        [
            "--dataset",
            "data/evaluation_dataset.json",
            "--judge",
            "groq",
        ]
    )

    assert report is not None
    assert report.judge == JudgeType.GROQ
    captured = capsys.readouterr().out
    assert "Judge: groq" in captured
    assert "Report written to:" not in captured


def test_run_cli_jev_signals_output(monkeypatch, tmp_path, capsys):
    from app import cli
    from app.schemas.evaluation_result import EvaluationVerdict, JudgeType

    mock_result = _make_mock_result(
        example_id="rag_001",
        judge=JudgeType.JEV,
        model="jev-latest",
        verdict=EvaluationVerdict.SUPPORTED,
        raw_output={
            "model": "jev-latest",
            "answers": {
                "hq_supported": {
                    "type": "noul",
                    "noul": 0.8842,
                }
            },
        },
    )

    monkeypatch.setattr(
        cli,
        "run_dataset_evaluation",
        lambda *args, **kwargs: {
            "total": 1,
            "successful": 1,
            "failed": 0,
            "results": [mock_result],
            "errors": [],
        },
    )

    output_file = tmp_path / "jev_report.json"
    cli.run_cli(
        [
            "--dataset",
            "data/evaluation_dataset.json",
            "--judge",
            "jev",
            "--output",
            str(output_file),
        ]
    )

    assert output_file.exists()
    captured = capsys.readouterr().out
    assert "Judge: jev" in captured
    assert "Jev signals:" in captured
    assert "rag_001: noul=0.8842, verdict=supported" in captured
    assert f"Report written to: {output_file}" in captured


def test_run_cli_raises_when_no_results_and_output_requested(monkeypatch, tmp_path):
    from app import cli

    monkeypatch.setattr(
        cli,
        "run_dataset_evaluation",
        lambda *args, **kwargs: {
            "total": 1,
            "successful": 0,
            "failed": 1,
            "results": [],
            "errors": [{"example_id": "rag_001", "message": "Failed"}],
        },
    )

    with pytest.raises(ValueError, match="Cannot export report"):
        cli.run_cli(
            [
                "--dataset",
                "data/evaluation_dataset.json",
                "--judge",
                "groq",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_run_cli_both_judges_exports_comparison_json(monkeypatch, tmp_path, capsys):
    from app import cli

    mock_groq = _make_mock_result(
        example_id="rag_001",
        judge=JudgeType.GROQ,
        model="groq-test-model",
        verdict=EvaluationVerdict.SUPPORTED,
    )
    mock_jev = _make_mock_result(
        example_id="rag_001",
        judge=JudgeType.JEV,
        model="jev-latest",
        verdict=EvaluationVerdict.SUPPORTED,
        raw_output={
            "model": "jev-latest",
            "answers": {
                "hq_supported": {
                    "type": "noul",
                    "noul": 0.95,
                }
            },
        },
    )

    monkeypatch.setattr(
        cli,
        "_run_jev_evaluation",
        lambda dataset_path: {
            "total": 1,
            "successful": 1,
            "failed": 0,
            "results": [mock_jev],
            "errors": [],
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_groq_evaluation",
        lambda dataset_path: {
            "total": 1,
            "successful": 1,
            "failed": 0,
            "results": [mock_groq],
            "errors": [],
        },
    )

    output_file = tmp_path / "comparison.json"
    comparison = cli.run_cli(
        [
            "--dataset",
            "data/evaluation_dataset.json",
            "--judge",
            "both",
            "--output",
            str(output_file),
            "--format",
            "json",
        ]
    )

    assert isinstance(comparison, ComparisonReport)
    assert output_file.exists()

    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert "comparison_id" in data
    assert len(data["reports"]) == 2
    assert data["reports"][0]["judge"] == "jev"
    assert data["reports"][1]["judge"] == "groq"

    captured = capsys.readouterr().out
    assert "Judge: jev" in captured
    assert "Judge: groq" in captured
    assert f"Report written to: {output_file}" in captured


def test_run_cli_both_judges_exports_comparison_csv(monkeypatch, tmp_path, capsys):
    import csv
    from app import cli

    mock_groq = _make_mock_result(
        example_id="rag_001",
        judge=JudgeType.GROQ,
        model="groq-test-model",
        verdict=EvaluationVerdict.SUPPORTED,
    )
    mock_jev = _make_mock_result(
        example_id="rag_001",
        judge=JudgeType.JEV,
        model="jev-latest",
        verdict=EvaluationVerdict.SUPPORTED,
        raw_output={
            "model": "jev-latest",
            "answers": {
                "hq_supported": {
                    "type": "noul",
                    "noul": 0.95,
                }
            },
        },
    )

    monkeypatch.setattr(
        cli,
        "_run_jev_evaluation",
        lambda dataset_path: {
            "total": 1,
            "successful": 1,
            "failed": 0,
            "results": [mock_jev],
            "errors": [],
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_groq_evaluation",
        lambda dataset_path: {
            "total": 1,
            "successful": 1,
            "failed": 0,
            "results": [mock_groq],
            "errors": [],
        },
    )

    output_file = tmp_path / "comparison.csv"
    cli.run_cli(
        [
            "--dataset",
            "data/evaluation_dataset.json",
            "--judge",
            "both",
            "--output",
            str(output_file),
            "--format",
            "csv",
        ]
    )

    assert output_file.exists()
    with output_file.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    judges = [row["judge"] for row in rows]
    assert "jev" in judges
    assert "groq" in judges

    captured = capsys.readouterr().out
    assert f"Report written to: {output_file}" in captured


def test_run_cli_both_judges_without_output(monkeypatch, capsys):
    from app import cli

    mock_groq = _make_mock_result(
        example_id="rag_001",
        judge=JudgeType.GROQ,
        model="groq-test-model",
        verdict=EvaluationVerdict.SUPPORTED,
    )
    mock_jev = _make_mock_result(
        example_id="rag_001",
        judge=JudgeType.JEV,
        model="jev-latest",
        verdict=EvaluationVerdict.SUPPORTED,
        raw_output={
            "model": "jev-latest",
            "answers": {
                "hq_supported": {
                    "type": "noul",
                    "noul": 0.95,
                }
            },
        },
    )

    monkeypatch.setattr(
        cli,
        "_run_jev_evaluation",
        lambda dataset_path: {
            "total": 1,
            "successful": 1,
            "failed": 0,
            "results": [mock_jev],
            "errors": [],
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_groq_evaluation",
        lambda dataset_path: {
            "total": 1,
            "successful": 1,
            "failed": 0,
            "results": [mock_groq],
            "errors": [],
        },
    )

    comparison = cli.run_cli(
        [
            "--dataset",
            "data/evaluation_dataset.json",
            "--judge",
            "both",
        ]
    )

    assert isinstance(comparison, ComparisonReport)
    captured = capsys.readouterr().out
    assert "Judge: jev" in captured
    assert "Judge: groq" in captured
    assert "Report written to:" not in captured


def test_run_cli_both_judges_raises_when_one_judge_fails(monkeypatch):
    from app import cli

    mock_jev = _make_mock_result(
        example_id="rag_001",
        judge=JudgeType.JEV,
        model="jev-latest",
        verdict=EvaluationVerdict.SUPPORTED,
        raw_output={
            "model": "jev-latest",
            "answers": {
                "hq_supported": {
                    "type": "noul",
                    "noul": 0.95,
                }
            },
        },
    )

    monkeypatch.setattr(
        cli,
        "_run_jev_evaluation",
        lambda dataset_path: {
            "total": 1,
            "successful": 1,
            "failed": 0,
            "results": [mock_jev],
            "errors": [],
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_groq_evaluation",
        lambda dataset_path: {
            "total": 1,
            "successful": 0,
            "failed": 1,
            "results": [],
            "errors": [{"example_id": "rag_001", "message": "Failed"}],
        },
    )

    with pytest.raises(
        ValueError, match="Cannot build comparison report"
    ):
        cli.run_cli(
            [
                "--dataset",
                "data/evaluation_dataset.json",
                "--judge",
                "both",
            ]
        )