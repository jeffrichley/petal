import os
from pathlib import Path

import pytest


@pytest.fixture
def mcp_server_script() -> str:
    """Fixture that provides the path to the MCP server script (not a test file)."""
    return str(Path(__file__).parent / "fixtures" / "mcp_server_script.py")


@pytest.fixture
def mcp_server_config(mcp_server_script: str) -> dict[str, dict[str, object]]:
    """Fixture that provides MCP server configuration for testing."""
    return {
        "test_server": {
            "command": "python",
            "args": [mcp_server_script],
            "transport": "stdio",
        }
    }


@pytest.fixture(autouse=True)
def fake_api_keys():
    """Automatically set fake API keys for all tests to prevent real API calls."""
    # Store original values
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    original_anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    original_google_key = os.environ.get("GOOGLE_API_KEY")
    original_cohere_key = os.environ.get("COHERE_API_KEY")
    original_huggingface_key = os.environ.get("HUGGINGFACE_API_KEY")

    # Set fake keys
    os.environ["OPENAI_API_KEY"] = "fake-openai-key-for-testing"
    os.environ["ANTHROPIC_API_KEY"] = "fake-anthropic-key-for-testing"
    os.environ["GOOGLE_API_KEY"] = "fake-google-key-for-testing"
    os.environ["COHERE_API_KEY"] = "fake-cohere-key-for-testing"
    os.environ["HUGGINGFACE_API_KEY"] = "fake-huggingface-key-for-testing"

    yield

    # Restore original values
    if original_openai_key is not None:
        os.environ["OPENAI_API_KEY"] = original_openai_key
    else:
        os.environ.pop("OPENAI_API_KEY", None)

    if original_anthropic_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = original_anthropic_key
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)

    if original_google_key is not None:
        os.environ["GOOGLE_API_KEY"] = original_google_key
    else:
        os.environ.pop("GOOGLE_API_KEY", None)

    if original_cohere_key is not None:
        os.environ["COHERE_API_KEY"] = original_cohere_key
    else:
        os.environ.pop("COHERE_API_KEY", None)

    if original_huggingface_key is not None:
        os.environ["HUGGINGFACE_API_KEY"] = original_huggingface_key
    else:
        os.environ.pop("HUGGINGFACE_API_KEY", None)


@pytest.fixture(autouse=True)
def disable_langsmith_tracing():
    """Disable LangSmith tracing for all tests to prevent warnings and real API calls."""
    # Store original values
    original_langsmith_tracing = os.environ.get("LANGSMITH_TRACING")
    original_langsmith_project = os.environ.get("LANGCHAIN_PROJECT")
    original_langsmith_endpoint = os.environ.get("LANGSMITH_ENDPOINT")

    # Set test values
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_PROJECT"] = "test-petal"
    os.environ["LANGSMITH_ENDPOINT"] = "http://localhost:8000"  # Fake endpoint

    yield

    # Restore original values
    if original_langsmith_tracing is not None:
        os.environ["LANGSMITH_TRACING"] = original_langsmith_tracing
    else:
        os.environ.pop("LANGSMITH_TRACING", None)

    if original_langsmith_project is not None:
        os.environ["LANGCHAIN_PROJECT"] = original_langsmith_project
    else:
        os.environ.pop("LANGCHAIN_PROJECT", None)

    if original_langsmith_endpoint is not None:
        os.environ["LANGSMITH_ENDPOINT"] = original_langsmith_endpoint
    else:
        os.environ.pop("LANGSMITH_ENDPOINT", None)
