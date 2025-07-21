"""Example demonstrating checkpointer conversation persistence with Petal agents."""

import asyncio
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from petal.core.config.checkpointer import CheckpointerConfig
from petal.core.factory import AgentFactory


class ConversationState(TypedDict):
    """State schema for conversation with checkpointer."""

    messages: Annotated[list, add_messages]
    name: str


async def demo_conversation_with_checkpointer():
    """Demonstrate conversation persistence with checkpointer."""
    print("=== Checkpointer Conversation Demo ===")

    # Create agent with memory checkpointer
    agent = await (
        AgentFactory(ConversationState)
        .with_chat(
            provider="openai",
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant that remembers information about the user.",
        )
        .with_checkpointer(CheckpointerConfig(type="memory"))
        .build()
    )

    print("✅ Agent built successfully with memory checkpointer")
    print("📝 Starting conversation...\n")

    # Ensure agent.graph is not None
    assert agent.graph is not None, "Agent graph should not be None"

    # Step 1: Tell the agent your name
    thread_id = "user-123"
    print("👤 Step 1: Telling the agent my name")
    result1 = await agent.graph.ainvoke(
        {"messages": [{"role": "user", "content": "My name is Alice."}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"🤖 Agent: {result1['messages'][-1].content}\n")

    # Step 2: Ask the agent to remember your name
    print("👤 Step 2: Asking the agent to remember my name")
    result2 = await agent.graph.ainvoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"🤖 Agent: {result2['messages'][-1].content}\n")

    # Step 3: Ask for more information
    print("👤 Step 3: Asking for more information")
    result3 = await agent.graph.ainvoke(
        {"messages": [{"role": "user", "content": "Tell me something about myself."}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"🤖 Agent: {result3['messages'][-1].content}\n")

    # Demonstrate state persistence
    print("📊 Checking conversation state...")
    current_state = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
    print(f"📈 Total messages in conversation: {len(current_state.values['messages'])}")

    # Get conversation history
    history = list(
        agent.graph.get_state_history({"configurable": {"thread_id": thread_id}})
    )
    print(f"🕒 Number of checkpoints saved: {len(history)}")

    return agent, thread_id


async def demo_new_conversation_same_user():
    """Demonstrate starting a new conversation with the same user."""
    print("\n=== New Conversation Demo (Same User) ===")

    # Create a new agent (simulating a new session)
    agent2 = await (
        AgentFactory(ConversationState)
        .with_chat(
            provider="openai",
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant that remembers information about the user.",
        )
        .with_checkpointer(CheckpointerConfig(type="memory"))
        .build()
    )

    # Ensure agent2.graph is not None
    assert agent2.graph is not None, "Agent graph should not be None"

    # Use the same thread_id to continue the conversation
    thread_id = "user-123"
    print("🔄 Starting new conversation with same thread_id...")

    result = await agent2.graph.ainvoke(
        {"messages": [{"role": "user", "content": "Do you remember my name?"}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"🤖 Agent: {result['messages'][-1].content}\n")


async def demo_different_users():
    """Demonstrate conversation isolation between different users."""
    print("\n=== Different Users Demo ===")

    agent3 = await (
        AgentFactory(ConversationState)
        .with_chat(
            provider="openai",
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant that remembers information about the user.",
        )
        .with_checkpointer(CheckpointerConfig(type="memory"))
        .build()
    )

    # Ensure agent3.graph is not None
    assert agent3.graph is not None, "Agent graph should not be None"

    # User 1
    print("👤 User 1 (Bob):")
    result_bob = await agent3.graph.ainvoke(
        {"messages": [{"role": "user", "content": "My name is Bob."}]},
        config={"configurable": {"thread_id": "bob-456"}},
    )
    print(f"🤖 Agent: {result_bob['messages'][-1].content}")

    # User 2 (should not know about Bob)
    print("\n👤 User 2 (Charlie):")
    result_charlie = await agent3.graph.ainvoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        config={"configurable": {"thread_id": "charlie-789"}},
    )
    print(f"🤖 Agent: {result_charlie['messages'][-1].content}")

    # User 1 again (should remember Bob)
    print("\n👤 User 1 (Bob) again:")
    result_bob2 = await agent3.graph.ainvoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        config={"configurable": {"thread_id": "bob-456"}},
    )
    print(f"🤖 Agent: {result_bob2['messages'][-1].content}")


async def demo_sqlite_persistence():
    """Demonstrate SQLite checkpointer for persistent storage."""
    print("\n=== SQLite Checkpointer Demo ===")

    try:
        # Create agent with SQLite checkpointer
        agent_sqlite = await (
            AgentFactory(ConversationState)
            .with_chat(
                provider="openai",
                model="gpt-4o-mini",
                system_prompt="You are a helpful assistant that remembers information about the user.",
            )
            .with_checkpointer(
                CheckpointerConfig(
                    type="sqlite", config={"db_file": "./data/conversations.db"}
                )
            )
            .build()
        )

        print("✅ Agent built successfully with SQLite checkpointer")

        # Ensure agent_sqlite.graph is not None
        assert agent_sqlite.graph is not None, "Agent graph should not be None"

        # Have a conversation
        thread_id = "sqlite-user-001"
        result = await agent_sqlite.graph.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": "My name is David and I like pizza."}
                ]
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        print(f"🤖 Agent: {result['messages'][-1].content}")

        # Ask about preferences
        result2 = await agent_sqlite.graph.ainvoke(
            {"messages": [{"role": "user", "content": "What do I like to eat?"}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        print(f"🤖 Agent: {result2['messages'][-1].content}")

        print("💾 Conversation saved to SQLite database")
        print("🔄 You can restart the program and the conversation will persist!")

    except Exception as e:
        print(f"❌ SQLite demo failed: {e}")
        print(
            "💡 To use SQLite checkpointers, install: pip install langgraph-checkpoint-sqlite"
        )


async def demo_state_inspection():
    """Demonstrate inspecting and manipulating conversation state."""
    print("\n=== State Inspection Demo ===")

    agent = await (
        AgentFactory(ConversationState)
        .with_chat(
            provider="openai",
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
        )
        .with_checkpointer(CheckpointerConfig(type="memory"))
        .build()
    )

    # Ensure agent.graph is not None
    assert agent.graph is not None, "Agent graph should not be None"

    thread_id = "inspect-demo"

    # Have a conversation
    await agent.graph.ainvoke(
        {"messages": [{"role": "user", "content": "Hello!"}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    await agent.graph.ainvoke(
        {"messages": [{"role": "user", "content": "How are you?"}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Inspect current state
    current_state = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
    print(f"📊 Current state values: {current_state.values}")
    print(f"📝 Total messages: {len(current_state.values['messages'])}")

    # Get full history
    history = list(
        agent.graph.get_state_history({"configurable": {"thread_id": thread_id}})
    )
    print(f"🕒 Checkpoints in history: {len(history)}")

    # Show checkpoint details
    for i, checkpoint in enumerate(history):
        print(f"  Checkpoint {i}: {checkpoint.config['configurable']['checkpoint_id']}")
        print(f"    Messages: {len(checkpoint.values['messages'])}")
        print(f"    Next: {checkpoint.next}")


async def main():
    """Run all checkpointer demos."""
    print("🚀 Petal Checkpointer Conversation Demo")
    print("=" * 50)

    # Demo 1: Basic conversation with memory checkpointer
    agent, thread_id = await demo_conversation_with_checkpointer()

    # Demo 2: New conversation with same user
    await demo_new_conversation_same_user()

    # Demo 3: Different users (conversation isolation)
    await demo_different_users()

    # Demo 4: SQLite persistence
    await demo_sqlite_persistence()

    # Demo 5: State inspection
    await demo_state_inspection()

    print("\n" + "=" * 50)
    print("✅ All demos completed!")
    print("\n💡 Key takeaways:")
    print("  • Use thread_id to maintain conversation context")
    print("  • Different thread_ids provide conversation isolation")
    print("  • Checkpointers automatically save state at each step")
    print("  • You can inspect and manipulate conversation state")
    print("  • SQLite checkpointers provide persistent storage")


if __name__ == "__main__":
    asyncio.run(main())
