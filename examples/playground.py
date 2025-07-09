#!/usr/bin/env python3
"""
Playground for Petal AgentBuilder: Demonstrates new agent building capabilities.
- Custom step agent
- LLM step agent (step-specific config)
- Multi-step chain (custom + LLM)

Note: Each step should return only the keys it modifies (partial state), not the full state dict.

Note: Currently, LLM steps require step-specific llm_config. The global with_llm() config
is not yet used by LLMStepStrategy.
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


# --- Example 1: Custom Step Agent ---
async def run_custom_step_agent():
    print("\n=== Custom Step Agent ===")

    # Return only the keys you want to update (partial state)
    async def process_step(state: dict) -> dict:  # noqa: ARG001
        return {"processed": True, "name": "processed_by_custom_step"}

    builder = AgentBuilder(TestState)
    agent = builder.with_step("custom", step_function=process_step).build()
    result = await agent.arun({"name": "test", "processed": False, "messages": []})
    print("Result:", result)


# --- Example 2: LLM Step Agent (Step-specific config) ---
async def run_llm_step_agent():
    print("\n=== LLM Step Agent (Step-specific config) ===")
    builder = AgentBuilder(TestState)
    agent = builder.with_step(
        "llm",
        prompt_template="Hello {name}, how are you?",
        llm_config={"provider": "openai", "model": "gpt-4o-mini"},
    ).build()
    result = await agent.arun({"name": "test", "processed": False, "messages": []})
    print("Result:", result)


# --- Example 3: Multi-Step Chain (Custom + LLM) ---
async def run_multi_step_chain():
    print("\n=== Multi-Step Chain (Custom + LLM) ===")

    # Each step returns only the keys it modifies
    async def add_greeting(state: dict) -> dict:
        return {"greeting": f"Hello {state['name']}!"}

    async def process_step(state: dict) -> dict:  # noqa: ARG001
        return {"processed": True}

    builder = AgentBuilder(TestState)
    agent = (
        builder.with_step("custom", step_function=add_greeting)
        .with_step("custom", step_function=process_step)
        .with_step(
            "llm",
            prompt_template="The user's name is {name}. Say something nice to them.",
            llm_config={"provider": "openai", "model": "gpt-4o-mini"},
        )
        .build()
    )
    result = await agent.arun({"name": "test", "processed": False, "messages": []})
    print("Result:", result)


async def main():
    await run_custom_step_agent()
    await run_llm_step_agent()
    await run_multi_step_chain()
    print("\n✅ Playground complete!")


if __name__ == "__main__":
    asyncio.run(main())
