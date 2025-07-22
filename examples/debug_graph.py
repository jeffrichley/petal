#!/usr/bin/env python3
"""
Debug script to explore graph object methods.
"""

from typing import TypedDict

from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from petal.core.factory import AgentFactory


class DemoState(TypedDict):
    """Demo state for testing."""

    messages: Annotated[list, add_messages]
    name: str


async def main():
    """Debug the graph object."""
    print("Creating agent factory...")

    # Create agent factory
    factory = AgentFactory(DemoState)

    # Add a simple LLM step
    factory.with_chat(
        llm_config={"provider": "openai", "model": "gpt-4"},
        system_prompt="You are a helpful assistant.",
    )

    # Build the agent
    agent = await factory.build()

    print(f"Agent built: {agent.built}")
    print(f"Agent graph type: {type(agent.graph)}")

    # Get the underlying graph
    if agent.graph is None:
        print("Agent graph is None")
        return

    graph_obj = agent.graph.get_graph()
    print(f"Graph object type: {type(graph_obj)}")
    print(f"Graph object dir: {dir(graph_obj)}")

    # Look for diagram methods
    diagram_methods = [
        attr
        for attr in dir(graph_obj)
        if "diagram" in attr.lower()
        or "draw" in attr.lower()
        or "mermaid" in attr.lower()
    ]
    print(f"Potential diagram methods: {diagram_methods}")

    # Check if there are any methods that might generate diagrams
    all_methods = [attr for attr in dir(graph_obj) if not attr.startswith("_")]
    print(f"All public methods: {all_methods}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
