from unittest.mock import Mock

import httpx
import pytest

from app.clients.jev_client import JevClient


def test_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(
        "app.clients.jev_client.settings.typesafe_api_key",
        "",
    )

    with pytest.raises(ValueError, match="API key is missing"):
        JevClient()


def test_evaluate_sends_correct_request():
    client = JevClient(
        api_key="test-api-key",
        model="jev-latest",
    )

    mock_response = Mock()
    mock_response.json.return_value = {
        "model": "jev-1.13.0",
        "answers": {
            "supported": {
                "type": "noul",
                "noul": 0.92,
            }
        },
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
        },
    }

    client._client.post = Mock(return_value=mock_response)

    result = client.evaluate(
        state="The provided context says the company was founded in London.",
        questions={
            "supported": {
                "type": "noul",
                "instructions": (
                    "Does the context explicitly support "
                    "the claim that the company is headquartered in London?"
                ),
            }
        },
    )

    client._client.post.assert_called_once_with(
        JevClient.BASE_URL,
        headers={
            "Authorization": "Bearer test-api-key",
            "Content-Type": "application/json",
        },
        json={
            "model": "jev-latest",
            "state": (
                "The provided context says the company was founded in London."
            ),
            "questions": {
                "supported": {
                    "type": "noul",
                    "instructions": (
                        "Does the context explicitly support "
                        "the claim that the company is headquartered in London?"
                    ),
                }
            },
        },
    )

    assert result["answers"]["supported"]["noul"] == 0.92
    assert result["model"] == "jev-1.13.0"

    client.close()


def test_evaluate_raises_for_http_error():
    client = JevClient(api_key="test-api-key")

    mock_response = Mock()
    mock_response.raise_for_status.side_effect = (
        httpx.HTTPStatusError(
            "Bad Request",
            request=httpx.Request("POST", JevClient.BASE_URL),
            response=httpx.Response(400),
        )
    )

    client._client.post = Mock(return_value=mock_response)

    with pytest.raises(httpx.HTTPStatusError):
        client.evaluate(
            state="Test state",
            questions={},
        )

    client.close()