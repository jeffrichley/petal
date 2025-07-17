#!/usr/bin/env python3
"""
Demo script for testing the diagram_graph functionality.
"""

from typing import TypedDict

from langgraph.graph.message import add_messages
from petal.core.factory import AgentFactory
from typing_extensions import Annotated


class DemoState(TypedDict):
    """Demo state for testing."""

    messages: Annotated[list, add_messages]
    name: str


async def main():
    """Test the diagram_graph functionality."""
    print("Creating agent factory...")

    # Create agent factory
    factory = AgentFactory(DemoState)

    # Add a simple LLM step
    factory.with_chat(
        llm_config={"provider": "openai", "model": "gpt-4"},
        system_prompt="You are a helpful assistant.",
    )

    # Generate diagram
    print("Generating diagram...")
    try:
        await factory.diagram_graph("agent_diagram.png")
        print("✅ Diagram saved to agent_diagram.png")
    except Exception as e:
        print(f"❌ Failed to generate diagram: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
