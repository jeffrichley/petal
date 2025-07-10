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

    # Set fake keys
    os.environ["OPENAI_API_KEY"] = "fake-openai-key-for-testing"
    os.environ["ANTHROPIC_API_KEY"] = "fake-anthropic-key-for-testing"

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
