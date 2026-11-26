from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    typesafe_api_key: str = ""
    jev_model: str = "jev-latest"

    # Optional threshold for interpreting Jev's hq_supported noul signal.
    #
    # This is a project-defined decision boundary, not a value provided by
    # TypeSafe or validated against an independent calibration dataset.
    #
    # When set:
    #   noul >= threshold  →  SUPPORTED
    #   noul <  threshold  →  INSUFFICIENT_EVIDENCE
    #
    # When not set (None):
    #   All Jev results receive an UNCERTAIN verdict.
    #
    # See app/evaluation/judge_normalizer.py for the full methodology note.
    jev_support_threshold: float | None = None

    groq_api_key: str = ""
    groq_model: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()