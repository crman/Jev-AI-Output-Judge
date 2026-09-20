from groq import Groq

from app.config import settings


class GroqClient:
    """Client for generating responses using Groq-hosted LLMs."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model

        if not self.api_key:
            raise ValueError(
                "Groq API key is missing. "
                "Set GROQ_API_KEY in your .env file."
            )

        if not self.model:
            raise ValueError(
                "Groq model is missing. "
                "Set GROQ_MODEL in your .env file."
            )

        self._client = Groq(api_key=self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a response using the configured Groq model.

        Returns:
            The generated assistant message as text.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content

        if content is None:
            raise ValueError("Groq returned an empty response.")

        return content

    def close(self) -> None:
        """Close the underlying Groq HTTP client."""
        self._client.close()