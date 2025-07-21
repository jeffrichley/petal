"""Example demonstrating checkpointer integration with Petal agents."""

import asyncio
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from petal.core.config.checkpointer import CheckpointerConfig
from petal.core.factory import AgentFactory


class DemoState(TypedDict):
    """Demo state schema."""

    messages: Annotated[list, add_messages]
    name: str


async def demo_memory_checkpointer():
    """Demonstrate memory checkpointer integration."""
    print("=== Memory Checkpointer Demo ===")

    # Create agent with memory checkpointer
    agent = (
        AgentFactory(DemoState)
        .with_chat(
            provider="openai",
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
        )
        .with_checkpointer(CheckpointerConfig(type="memory"))
        .build()
    )

    # Build the agent
    built_agent = await agent

    print(f"Agent built successfully with checkpointer: {built_agent}")
    print("Memory checkpointer will store state in memory during execution")
    print()


async def demo_disabled_checkpointer():
    """Demonstrate disabled checkpointer (default behavior)."""
    print("=== Disabled Checkpointer Demo ===")

    # Create agent without checkpointer (default)
    agent = (
        AgentFactory(DemoState)
        .with_chat(
            provider="openai",
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
        )
        .build()
    )

    # Build the agent
    built_agent = await agent

    print(f"Agent built successfully without checkpointer: {built_agent}")
    print("No state persistence - each run starts fresh")
    print()


async def demo_postgres_checkpointer():
    """Demonstrate postgres checkpointer configuration."""
    print("=== Postgres Checkpointer Demo ===")

    try:
        # Create agent with postgres checkpointer
        postgres_config = CheckpointerConfig(
            type="postgres",
            config={"connection_string": "postgresql://user:pass@localhost:5432/petal"},
        )

        agent = (
            AgentFactory(DemoState)
            .with_chat(
                provider="openai",
                model="gpt-4o-mini",
                system_prompt="You are a helpful assistant.",
            )
            .with_checkpointer(postgres_config)
            .build()
        )

        # Build the agent
        built_agent = await agent

        print(f"Agent built successfully with postgres checkpointer: {built_agent}")
        print("Postgres checkpointer will persist state to database")
        print()
    except ImportError as e:
        print(
            "Postgres checkpointer demo skipped - postgres dependencies not installed"
        )
        print(f"Error: {e}")
        print("To use postgres checkpointers, install: pip install psycopg2-binary")
        print()
    except Exception as e:
        print(f"Postgres checkpointer demo failed: {e}")
        print()


def with_checkpointer(checkpointer_config: CheckpointerConfig):
    """Helper method to add checkpointer to AgentFactory."""
    # This would be added to AgentFactory
    print(f"Checkpointer config: {checkpointer_config.type}")
    pass


async def main():
    """Run all checkpointer demos."""
    print("Petal Checkpointer Integration Demo")
    print("=" * 40)
    print()

    await demo_memory_checkpointer()
    await demo_disabled_checkpointer()
    await demo_postgres_checkpointer()

    # Run the consistency demonstration (synchronous)
    demonstrate_consistent_interface()

    print("Demo completed!")
    print()
    print("Note: This demo shows configuration only.")
    print("Actual execution would require proper API keys and database setup.")


def demonstrate_consistent_interface():
    """Demonstrate the consistent with_checkpointer interface in both AgentFactory and AgentBuilder."""
    print("\n" + "=" * 60)
    print("CONSISTENT CHECKPOINTER INTERFACE DEMONSTRATION")
    print("=" * 60)

    # Example 1: Using AgentFactory with_checkpointer
    print("\n1. AgentFactory with_checkpointer:")
    from petal.core.factory import AgentFactory
    from typing_extensions import TypedDict

    class DemoState(TypedDict):
        messages: list
        counter: int

    # AgentFactory approach
    factory_agent = (
        AgentFactory(DemoState)
        .with_chat(provider="openai", model="gpt-4o-mini")
        .with_checkpointer(CheckpointerConfig(type="memory", enabled=True))
        .add(lambda state: {"counter": state.get("counter", 0) + 1})
    )

    print(
        f"   ✓ AgentFactory.with_checkpointer() works - {type(factory_agent).__name__}"
    )

    # Example 2: Using AgentBuilder with_checkpointer
    print("\n2. AgentBuilder with_checkpointer:")
    from petal.core.builders.agent import AgentBuilder

    # AgentBuilder approach
    builder_agent = (
        AgentBuilder(DemoState)
        .with_llm(provider="openai", model="gpt-4o-mini")
        .with_checkpointer(CheckpointerConfig(type="memory", enabled=True))
        .with_step(
            "custom",
            step_function=lambda state: {"counter": state.get("counter", 0) + 1},
        )
    )

    print(
        f"   ✓ AgentBuilder.with_checkpointer() works - {type(builder_agent).__name__}"
    )

    print("\n✅ Both interfaces are now consistent!")
    print("   - AgentFactory.with_checkpointer()")
    print("   - AgentBuilder.with_checkpointer()")
    print("   - Both delegate to AgentConfig.set_checkpointer()")
    print("   - Both support the same CheckpointerConfig types")


if __name__ == "__main__":
    asyncio.run(main())
