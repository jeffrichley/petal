import asyncio
from typing import Annotated

from langgraph.graph.message import add_messages
from petal.core.factory import AgentFactory
from typing_extensions import TypedDict

# from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import InMemorySaver

# agent = create_react_agent()


async def main():

    second_topic = "bananas"

    # Define steps that prepare data for the LLM
    def step1(state):
        # state["topic"] = state.get("topic", "penguins")
        # state["mood"] = "silly"
        # return state

        return {
            "topic": state.get("topic", "penguins"),
            "mood": "silly",
        }

    def step2(state):
        # Extract the latest AI message content from messages
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

    def step3(state):
        # state["final_topic"] = second_topic
        # return state

        return {
            "final_topic": second_topic,
        }

    print("=== Building Agent with Multiple LLM Steps ===")

    class MyState(TypedDict):
        messages: Annotated[list, add_messages]
        processed_response: str
        final_topic: str
        topic: str
        mood: str

    # Build the agent with chained methods - each LLM step has its own prompts
    agent = (
        AgentFactory(MyState)
        .add(step1)
        .with_chat()  # First LLM step
        .with_prompt("Tell me a {mood} joke about {topic}.")
        .with_system_prompt("You are a helpful AI comedian.")
        .add(step2)
        .add(step3)
        .with_chat()  # Second LLM step
        .with_prompt("Now tell me a serious fact about {final_topic}.")
        .with_system_prompt("You are a knowledgeable teacher.")
        .build()
    )

    # Prepare the initial state
    state = {"topic": "bananas"}

    print("=== Synchronous Run ===")
    # Run the agent - each LLM step will be executed in order
    result = agent.run(state)
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

    print("\n=== Asynchronous Run ===")

    state = {"topic": "fajitas"}
    second_topic = "fajitas"

    # Run the agent asynchronously
    async_result = await agent.arun(state)
    print(f"Async Messages: {len(async_result.get('messages', []))} messages")
    if async_result.get("messages"):
        for i, msg in enumerate(async_result["messages"]):
            print(
                f"  Message {i}: {type(msg).__name__} - {getattr(msg, 'content', str(msg))}"
            )


if __name__ == "__main__":
    asyncio.run(main())
