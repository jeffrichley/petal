from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated


class DefaultState(TypedDict):
    """Default state schema for agents."""

    messages: Annotated[list, add_messages]


def merge_state(old_state: dict, updates: dict) -> dict:
    """Merge updates into the old state."""
    return {**old_state, **updates}


def step1(state):
    print(f"Step1 received state: {state}")
    result = {"x": 1}
    print(f"Step1 returning: {result}")
    return result


def step2(state):
    print(f"Step2 received state: {state}")
    result = {"x": state["x"] + 2}
    print(f"Step2 returning: {result}")
    return result


# Create the graph exactly like our factory
graph_builder = StateGraph(DefaultState)
graph_builder.set_state_reducer(merge_state)
graph_builder.add_node("step_0", step1)
graph_builder.add_node("step_1", step2)
graph_builder.add_edge(START, "step_0")
graph_builder.add_edge("step_0", "step_1")
graph_builder.add_edge("step_1", END)

graph = graph_builder.compile()

# Test it
print("Testing with empty state:")
result = graph.invoke({})
print("Final result:", result)
