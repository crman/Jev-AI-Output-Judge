from pydantic import BaseModel, Field


class GroundTruth(BaseModel):
    contradiction: bool
    unsupported_claim: bool


class EvaluationExample(BaseModel):
    id: str = Field(..., description="Unique evaluation example ID")

    question: str
    context: str
    answer: str

    ground_truth: GroundTruth

    category: str = Field(
        ...,
        description="Type of evaluation scenario",
    )


class EvaluationDataset(BaseModel):
    dataset_name: str
    version: str
    examples: list[EvaluationExample]