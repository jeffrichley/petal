from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TestState(TypedDict):
    x: int
    processed: bool
    messages: list


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


def step3(state):
    print(f"Step3 received state: {state}")
    result = {"processed": True}
    print(f"Step3 returning: {result}")
    return result


# Create the graph
graph_builder = StateGraph(TestState)
graph_builder.add_node("step1", step1)
graph_builder.add_node("step2", step2)
graph_builder.add_node("step3", step3)
graph_builder.add_edge(START, "step1")
graph_builder.add_edge("step1", "step2")
graph_builder.add_edge("step2", "step3")
graph_builder.add_edge("step3", END)

graph = graph_builder.compile()

# Test it
print("Testing with empty state:")
result = graph.invoke({})
print("Final result:", result)

print("\nTesting with initial state:")
result = graph.invoke({"messages": [{"role": "user", "content": "test"}]})
print("Final result:", result)
