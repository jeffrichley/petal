"""
Demonstration of ConfigDiscovery functionality.

This example shows how to use the ConfigDiscovery strategy to automatically
discover tools from YAML configuration files.
"""

import os
import tempfile

import yaml
from langchain_core.tools import tool

from petal.core.discovery.config import ConfigDiscovery
from petal.core.registry import ToolRegistry


# Create a sample tool for demonstration
@tool
def sample_echo(text: str) -> str:
    """Echo the input text."""
    return f"Echo: {text}"


@tool
def sample_math_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def main():
    """Demonstrate ConfigDiscovery functionality."""

    # Create a temporary YAML config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "tools": {
                "echo_tool": {"module": "__main__", "function": "sample_echo"},
                "math_tool": {"module": "__main__", "function": "sample_math_add"},
            }
        }
        yaml.dump(config, f)
        config_path = f.name

    try:
        # Create ConfigDiscovery with our config file
        config_discovery = ConfigDiscovery(config_locations=[config_path])

        # Create ToolRegistry and add the discovery strategy
        registry = ToolRegistry()
        registry.add_discovery_strategy(config_discovery)

        print("🔍 Discovering tools from config...")

        # Discover tools by name
        echo_tool = await registry.resolve("echo_tool")
        math_tool = await registry.resolve("math_tool")

        print(f"✅ Found echo_tool: {echo_tool.name}")
        print(f"✅ Found math_tool: {math_tool.name}")

        # Test the tools
        echo_result = await echo_tool.ainvoke({"text": "Hello, ConfigDiscovery!"})
        math_result = await math_tool.ainvoke({"a": 5, "b": 3})

        print(f"📤 Echo tool result: {echo_result}")
        print(f"📤 Math tool result: {math_result}")

        # List all registered tools
        print(f"📋 All registered tools: {registry.list()}")

    finally:
        # Clean up temporary file
        os.unlink(config_path)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
