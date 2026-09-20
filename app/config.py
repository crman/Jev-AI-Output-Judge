from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    typesafe_api_key: str = ""
    jev_model: str = "jev-latest"

    groq_api_key: str = ""
    groq_model: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()