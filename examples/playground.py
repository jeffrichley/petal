#!/usr/bin/env python3
"""
Simple playground demonstrating with_step and with_llm methods.
"""

import asyncio
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from petal.core.builders.agent import AgentBuilder


# Define a state type for demonstration
class TestState(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    processed: bool
    personality: str


async def run_example():
    print("=== Simple with_step and with_llm Example ===")

    # Custom step function
    async def custom_step(state: dict) -> dict:
        return {
            "processed": True,
            "custom_result": f"Hello {state['name']}!",
            "personality": "pirate",
        }

    # Build agent with custom step and LLM step
    builder = AgentBuilder(TestState)
    agent = (
        builder.with_step("custom", step_function=custom_step)
        .with_step(
            "llm",
            prompt_template="The user's name is {name}. Say something nice to them.",
        )
        .with_system_prompt(
            "You are a helpful assistant that talks like a {personality}."
        )
        # .with_step(
        #     "llm",
        #     prompt_template="The user's name is {name}. Say something nice to them.",
        #     system_prompt="You are a helpful assistant that talks like a {personality}.",
        # )
        .with_llm(provider="openai", model="gpt-4o-mini")
        .build()
    )

    # Run the agent
    result = await agent.arun({"name": "test", "processed": False, "messages": []})
    print("Result:", result)
    print("AI response:", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(run_example())
