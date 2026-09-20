from unittest.mock import Mock

import pytest

from app.clients.groq_client import GroqClient


def test_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(
        "app.clients.groq_client.settings.groq_api_key",
        "",
    )

    with pytest.raises(ValueError, match="Groq API key is missing"):
        GroqClient(model="test-model")


def test_client_requires_model(monkeypatch):
    monkeypatch.setattr(
        "app.clients.groq_client.settings.groq_model",
        "",
    )

    with pytest.raises(ValueError, match="Groq model is missing"):
        GroqClient(api_key="test-api-key")


def test_generate_returns_assistant_content(monkeypatch):
    mock_sdk = Mock()
    mock_response = Mock()
    mock_response.choices = [
        Mock(message=Mock(content="The answer is supported."))
    ]
    mock_sdk.chat.completions.create.return_value = mock_response

    monkeypatch.setattr(
        "app.clients.groq_client.Groq",
        Mock(return_value=mock_sdk),
    )

    client = GroqClient(
        api_key="test-api-key",
        model="test-model",
    )

    result = client.generate(
        system_prompt="You are an evaluator.",
        user_prompt="Is this answer supported?",
    )

    assert result == "The answer is supported."

    mock_sdk.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[
            {"role": "system", "content": "You are an evaluator."},
            {"role": "user", "content": "Is this answer supported?"},
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    client.close()


def test_generate_raises_when_content_is_none(monkeypatch):
    mock_sdk = Mock()
    mock_response = Mock()
    mock_response.choices = [
        Mock(message=Mock(content=None))
    ]
    mock_sdk.chat.completions.create.return_value = mock_response

    monkeypatch.setattr(
        "app.clients.groq_client.Groq",
        Mock(return_value=mock_sdk),
    )

    client = GroqClient(
        api_key="test-api-key",
        model="test-model",
    )

    with pytest.raises(ValueError, match="Groq returned an empty response"):
        client.generate(
            system_prompt="You are an evaluator.",
            user_prompt="Evaluate this answer.",
        )

    client.close()