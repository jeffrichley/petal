from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TestState(TypedDict):
    x: int
    messages: list


def step1(state):
    return {"x": 1}


def step2(state):
    return {"x": state["x"] + 2}


# Create the graph
graph_builder = StateGraph(TestState)
graph_builder.add_node("step1", step1)
graph_builder.add_node("step2", step2)
graph_builder.add_edge(START, "step1")
graph_builder.add_edge("step1", "step2")
graph_builder.add_edge("step2", END)

graph = graph_builder.compile()

# Test it
result = graph.invoke({})
print("Result:", result)
