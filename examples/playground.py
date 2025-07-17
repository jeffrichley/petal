#!/usr/bin/env python3
"""
Simple playground demonstrating with_step, with_llm, and YAML loading methods.
"""

import asyncio
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from petal.core.builders.agent import AgentBuilder
from petal.core.factory import AgentFactory


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
        await (
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
    )

    # Run the agent
    result = await agent.arun({"name": "test", "processed": False, "messages": []})
    print("Result:", result)
    print("AI response:", result["messages"][-1].content)


async def run_yaml_example():
    print("\n=== YAML Node Loading Example ===")

    # Create factory
    factory = AgentFactory(TestState)

    # Load node from YAML (if the file exists)
    try:
        await factory.node_from_yaml("examples/yaml/llm_node.yaml")
        print("✓ Successfully loaded LLM node from YAML")

        # Build agent
        agent = await factory.build()
        print("✓ Successfully built agent with YAML node")

        # Run the agent
        result = await agent.arun({"name": "test", "processed": False, "messages": []})
        print("✓ Successfully ran agent with YAML node")
        print("AI response:", result["messages"][-1].content)

    except FileNotFoundError:
        print("⚠ YAML file not found. Make sure examples/yaml/llm_node.yaml exists.")
    except Exception as e:
        print(f"⚠ Error loading YAML node: {e}")


if __name__ == "__main__":
    asyncio.run(run_example())
    asyncio.run(run_yaml_example())
