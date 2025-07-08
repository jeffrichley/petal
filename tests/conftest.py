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
