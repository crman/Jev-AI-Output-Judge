import httpx

from app.config import settings


class JevClient:
    """HTTP client for the TypeSafe Jev API."""

    BASE_URL = "https://api.typesafe.ai/v1/systemone"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or settings.typesafe_api_key
        self.model = model or settings.jev_model
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "TypeSafe API key is missing. "
                "Set TYPESAFE_API_KEY in your .env file."
            )

        self._client = httpx.Client(timeout=self.timeout)

    def evaluate(
        self,
        state: str | dict | list,
        questions: dict,
    ) -> dict:
        """
        Submit a structured evaluation request to Jev.

        Args:
            state: Content Jev should evaluate.
            questions: Named Jev questions and their definitions.

        Returns:
            Parsed JSON response from the Jev API.

        Raises:
            httpx.HTTPStatusError: If the API returns an HTTP error.
            httpx.RequestError: If the request fails at the transport level.
            ValueError: If the response is not valid JSON.
        """
        payload = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }

        response = self._client.post(
            self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()