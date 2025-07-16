#!/usr/bin/env python3
"""Debug script for folder discovery."""

import asyncio
import tempfile
from pathlib import Path

from langchain_core.tools import BaseTool
from petal.core.discovery.folder import FolderDiscovery
from petal.core.registry import ToolRegistry


async def debug_folder_discovery():
    """Debug folder discovery functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_file = Path(temp_dir) / "test_tool.py"

        test_file.write_text(
            """
from petal.core.decorators import petaltool

@petaltool("test_tool")
def test_tool():
    \"\"\"A test tool for debugging.\"\"\"
    pass
"""
        )

        print(f"Created test file: {test_file}")
        print(f"File contents: {test_file.read_text()}")

        discovery = FolderDiscovery(
            folders=[temp_dir], exclude_patterns=["*_temp.py"], auto_discover=False
        )

        print(f"Discovery folders: {discovery.folders}")
        print(f"Discovery exclude patterns: {discovery.exclude_patterns}")

        # Debug the scanning process
        print("\n=== Debugging scanning process ===")

        # Check if folder exists
        folder_path = Path(temp_dir)
        print(f"Folder exists: {folder_path.exists()}")

        # Find Python files
        python_files = list(folder_path.glob("*.py"))
        print(f"Python files found: {python_files}")

        for file_path in python_files:
            print(f"\nProcessing file: {file_path}")

            # Check if file should be excluded
            should_exclude = discovery._should_exclude_file(file_path)
            print(f"Should exclude: {should_exclude}")

            if not should_exclude:
                # Debug registry state
                registry = ToolRegistry()
                print(f"Registry tools before import: {registry.list()}")

                # Try to load tools from file
                file_tools = await discovery._load_tools_from_file(file_path)
                print(f"Tools found in file: {list(file_tools.keys())}")

                print(f"Registry tools after import: {registry.list()}")

                # Diagnostic: Print all names and types in the module
                print("\n=== Module namespace diagnostic ===")
                import importlib.util
                import sys

                module_name = f"discovered_{file_path.stem}"
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    module.__package__ = "petal.core.discovery"
                    module.__loader__ = spec.loader
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    print(f"Module name: {module_name}")
                    print("All names in module:")
                    for name, obj in vars(module).items():
                        if not name.startswith("__"):
                            print(f"  {name}: {type(obj)} = {obj}")

                    # Specifically check test_tool
                    if hasattr(module, "test_tool"):
                        test_tool_obj = module.test_tool
                        print("\nSpecific test_tool check:")
                        print(f"  test_tool: {type(test_tool_obj)} = {test_tool_obj}")
                        print(f"  Is BaseTool: {isinstance(test_tool_obj, BaseTool)}")
                    else:
                        print("\ntest_tool not found in module namespace")

        # Try to discover the tool
        tool = await discovery.discover("test_tool")
        print(f"\nDiscovered tool: {tool}")

        # Check what tools were found
        all_tools = await discovery.discover_all()
        print(f"All discovered tools: {list(all_tools.keys())}")


if __name__ == "__main__":
    asyncio.run(debug_folder_discovery())
