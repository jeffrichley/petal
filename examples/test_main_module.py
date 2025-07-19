#!/usr/bin/env python3
"""
Example demonstrating how @petaltool works with __main__ module.

This script shows how tools defined in the main module (when running a script directly)
get special handling - they use just the function name without namespace.

Run this script directly to see the behavior:
    python examples/test_main_module.py
"""

from petal.core.decorators import petaltool
from petal.core.registry import ToolRegistry


@petaltool
def main_tool(query: str) -> str:
    """A tool defined in the main module."""
    return f"Main tool says: {query}"


@petaltool("custom_name")
def custom_named_tool(query: str) -> str:
    """A tool with custom name in the main module."""
    return f"Custom tool says: {query}"


@petaltool("namespace:custom:name")
def fully_qualified_tool(query: str) -> str:
    """A tool with fully qualified name in the main module."""
    return f"Fully qualified tool says: {query}"


def main():
    """Demonstrate the __main__ module behavior."""
    print("=== Testing @petaltool with __main__ module ===\n")

    # Check the registry to see how tools were registered
    registry = ToolRegistry()
    registered_tools = registry.list()

    print("Registered tools:")
    for tool_name in registered_tools:
        print(f"  - {tool_name}")

    print("\n=== Testing tool invocations ===")

    # Test each tool
    tools_to_test = [
        ("main_tool", "Hello from main"),
        ("custom_name", "Hello from custom"),
        ("namespace:custom:name", "Hello from fully qualified"),
    ]

    for tool_name, test_input in tools_to_test:
        try:
            tool = registry._registry[tool_name]
            result = tool.invoke({"query": test_input})
            print(f"✓ {tool_name}: {result}")
        except Exception as e:
            print(f"✗ {tool_name}: Error - {e}")

    print("\n=== Key Points ===")
    print("1. Tools in __main__ module use just function name (no namespace)")
    print("2. Custom names without colons also get no namespace")
    print("3. Fully qualified names (with colons) preserve the full name")
    print("4. This prevents namespace pollution when running scripts directly")


if __name__ == "__main__":
    main()
