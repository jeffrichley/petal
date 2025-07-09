from typing import Annotated

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]
    x: int


graph_builder = StateGraph(State)


llm = init_chat_model("openai:gpt-4o-mini")


def non_chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def last_node(state: State):  # noqa: ARG001
    return {"x": 42}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("non_chatbot", non_chatbot)
graph_builder.add_node("last_node", last_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "non_chatbot")
graph_builder.add_edge("non_chatbot", "last_node")
graph_builder.add_edge("last_node", END)
graph = graph_builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
print(result)
