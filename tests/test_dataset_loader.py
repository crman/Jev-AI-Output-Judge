from pathlib import Path

import pytest

from app.evaluation.dataset_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_dataset.json"


def test_load_evaluation_dataset():
    dataset = load_dataset(DATASET_PATH)

    assert dataset.dataset_name == "jevjudge_rag_evaluation"
    assert len(dataset.examples) == 10


def test_dataset_examples_have_unique_ids():
    dataset = load_dataset(DATASET_PATH)

    ids = [example.id for example in dataset.examples]

    assert len(ids) == len(set(ids))


def test_missing_dataset_raises_error():
    with pytest.raises(FileNotFoundError):
        load_dataset("data/missing_dataset.json")


def test_ground_truth_labels_are_boolean():
    dataset = load_dataset(DATASET_PATH)

    for example in dataset.examples:
        assert isinstance(
            example.ground_truth.contradiction, bool
        )
        assert isinstance(
            example.ground_truth.unsupported_claim, bool
        )