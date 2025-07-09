from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from petal.core.factory import AgentFactory
from typing_extensions import TypedDict


class DebugState(TypedDict):
    messages: list
    step1_result: str
    step2_result: str


def step1(state):
    print(f"Step1 received: {state}")
    result = {"step": 1}
    print(f"Step1 returning: {result}")
    return result


def step2(state):
    print(f"Step2 received: {state}")
    result = {"processed": True}
    print(f"Step2 returning: {result}")
    return result


with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
    mock_llm = Mock()
    mock_response = AIMessage(content="Test response")
    mock_llm.invoke.return_value = mock_response
    mock_chat_openai.return_value = mock_llm

    agent = AgentFactory(DebugState).add(step1).add(step2).with_chat().build()

    print("Running agent...")
    result = agent.run({"messages": [HumanMessage(content="test")]})
    print(f"Final result: {result}")
