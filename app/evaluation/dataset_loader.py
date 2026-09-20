import json
from pathlib import Path

from app.schemas.evaluation import EvaluationDataset


def load_dataset(file_path: str | Path) -> EvaluationDataset:
    """
    Load and validate an evaluation dataset from a JSON file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return EvaluationDataset.model_validate(data)