import os
from unittest.mock import patch

import pytest
from langchain_core.tools import BaseTool

from petal.core.discovery.folder import FolderDiscovery


@pytest.mark.asyncio
async def test_folder_discovery_finds_tool(tmp_path):
    tool_file = tmp_path / "tool_mod.py"
    tool_file.write_text(
        "from petal.core.decorators import petaltool\n" 
        "@petaltool\n" 
        "def sample_tool():\n" 
        "    \"Return text\"\n" 
        "    return 'found'\n"
    )

    discovery = FolderDiscovery(folders=[str(tmp_path)])
    with patch("petal.core.registry.ToolRegistry.add", return_value=None):
        tool = await discovery.discover("sample_tool")

    assert isinstance(tool, BaseTool)
    assert tool.name == "sample_tool"


@pytest.mark.asyncio
async def test_folder_discovery_handles_invalid_files(tmp_path):
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("def invalid: pass")

    discovery = FolderDiscovery(folders=[str(tmp_path)])
    with patch("petal.core.registry.ToolRegistry.add", return_value=None):
        tool = await discovery.discover("nonexistent")

    assert tool is None


@pytest.mark.asyncio
async def test_folder_discovery_missing_folder(tmp_path):
    missing_folder = tmp_path / "missing"
    discovery = FolderDiscovery(folders=[str(missing_folder)])
    tool = await discovery.discover("any")
    assert tool is None
