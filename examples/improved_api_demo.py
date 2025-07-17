#!/usr/bin/env python3
"""
Demonstration of the improved with_llm API using named parameters.
"""

import asyncio
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from petal.core.builders.agent import AgentBuilder


# Define a state type for demonstration
class DemoState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    response: str


async def run_demo():
    print("=== Improved with_llm API Demo ===")
    print("Using named parameters instead of magic dictionary keys!")
    print()

    # Build agent with the new, more Pythonic API
    builder = AgentBuilder(DemoState)
    agent = await (
        builder.with_step(
            "llm", prompt_template="User says: {user_input}. Respond helpfully."
        )
        .with_llm(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,  # More creative responses
            max_tokens=150,  # Shorter responses
        )
        .build()
    )

    # Run the agent
    result = await agent.arun(
        {"user_input": "Hello! How are you today?", "response": "", "messages": []}
    )

    print("Agent Response:", result["messages"][-1].content)
    print()
    print("✅ Much cleaner API! No more guessing dictionary keys.")


if __name__ == "__main__":
    asyncio.run(run_demo())
