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
        # Use the LLM response from the first chat step
        # state["processed_response"] = (
        #     f"Processed: {state.get('llm_response', 'No response')}"
        # )
        # return state

        return {
            "processed_response": f"Processed: {state.get('llm_response', 'No response')}",
        }

    def step3(state):  # noqa: ARG001
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
    agent = AgentFactory(MyState).add(step1).add(step2).add(step3).build()

    # Prepare the initial state
    state = {"topic": "boogies"}

    print("=== Synchronous Run ===")
    # Run the agent - each LLM step will be executed in order
    result = agent.run(state)
    print(f"Step 1 - Topic: {result['topic']}")
    print(f"Step 1 - Mood: {result['mood']}")
    print(f"Step 1 - LLM Response: {result.get('llm_response', 'No response')}")
    print(f"Step 2 - Final Topic: {result.get('final_topic', 'No final topic')}")
    print(f"Step 3 - Processed: {result.get('processed_response', 'No processed')}")


if __name__ == "__main__":
    asyncio.run(main())
