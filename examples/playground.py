import asyncio
from typing import Annotated

from langgraph.graph.message import add_messages
from petal.core.factory import AgentFactory
from typing_extensions import TypedDict

# from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import InMemorySaver

# agent = create_react_agent()


async def step1(state):
    return {
        "topic": state.get("topic", "penguins"),
        "mood": "silly",
    }


async def step2(state):
    messages = state.get("messages", [])
    ai_content = None
    for msg in reversed(messages):
        if type(msg).__name__ == "AIMessage":
            ai_content = getattr(msg, "content", None)
            break
    if ai_content is None:
        ai_content = "No AI message found"
    return {
        "processed_response": f"Processed: {ai_content}",
    }


def create_step3(second_topic: str):
    async def step3(_state):
        return {
            "final_topic": second_topic,
        }

    return step3


def build_agent(second_topic: str):
    """Build the agent with multiple LLM steps."""

    class MyState(TypedDict):
        messages: Annotated[list, add_messages]
        processed_response: str
        final_topic: str
        topic: str
        mood: str

    return (
        AgentFactory(MyState)
        .add(step1)
        .with_chat()  # First LLM step
        .with_prompt("Tell me a {mood} joke about {topic}.")
        .with_system_prompt("You are a helpful AI comedian.")
        .add(step2)
        .add(create_step3(second_topic))
        .with_chat()  # Second LLM step
        .with_prompt("Now tell me a serious fact about {final_topic}.")
        .with_system_prompt("You are a knowledgeable teacher.")
        .build()
    )


def print_results(result, title: str):
    """Print the results of agent execution."""
    print(f"=== {title} ===")
    print(f"Step 1 - Topic: {result['topic']}")
    print(f"Step 1 - Mood: {result['mood']}")
    print(f"Step 1 - Messages: {len(result.get('messages', []))} messages")
    if result.get("messages"):
        for i, msg in enumerate(result["messages"]):
            print(
                f"  Message {i}: {type(msg).__name__} - {getattr(msg, 'content', str(msg))}"
            )
    print(f"Step 2 - Final Topic: {result.get('final_topic', '** No final topic **')}")
    print(
        f"Step 3 - Processed: {result.get('processed_response', '** No processed **')}"
    )


async def run_sync_test(agent, state: dict):
    """Run the synchronous test."""
    result = await agent.arun(state)
    print_results(result, "Synchronous Run")


async def run_async_test(agent, state: dict):
    """Run the asynchronous test."""
    async_result = await agent.arun(state)
    print("=== Asynchronous Run ===")
    print(f"Async Messages: {len(async_result.get('messages', []))} messages")
    if async_result.get("messages"):
        for i, msg in enumerate(async_result["messages"]):
            print(
                f"  Message {i}: {type(msg).__name__} - {getattr(msg, 'content', str(msg))}"
            )


async def main():
    """Main function with reduced complexity."""
    second_topic = "bananas"

    # Build the agent
    agent = build_agent(second_topic)

    # Run synchronous test
    state = {"topic": "bananas"}
    await run_sync_test(agent, state)

    # Run asynchronous test
    state = {"topic": "fajitas"}
    await run_async_test(agent, state)


if __name__ == "__main__":
    asyncio.run(main())
