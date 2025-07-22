from typing import Annotated, get_type_hints
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from tests.petal.conftest_factory import ChatState, DefaultState
from typing_extensions import TypedDict

from petal.core.factory import AgentFactory


@pytest.mark.asyncio
async def test_dynamic_state_type_cache():
    """Test that the dynamic state type cache works properly."""
    from typing_extensions import TypedDict

    class CustomState(TypedDict):
        custom_field: str

    # Create two agents with the same state type and chat
    agent1 = await AgentFactory(CustomState).with_chat().build()
    agent2 = await AgentFactory(CustomState).with_chat().build()

    # Check that both agents use the same state type (cached)
    assert agent1.state_type == agent2.state_type
    assert agent1.state_type is not None
    assert agent1.state_type.__name__ == "CustomStateWithMessagesAddedByPetal"

    # Verify the state type has both the original field and messages
    type_hints = get_type_hints(agent1.state_type, include_extras=True)
    assert "custom_field" in type_hints
    assert "messages" in type_hints

    # Test that the agents work correctly
    async def step(_state):
        return {"custom_field": "updated"}

    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent3 = await AgentFactory(CustomState).add(step).with_chat().build()
        result = await agent3.arun({"custom_field": "initial"})

        assert "messages" in result
        assert "custom_field" in result


@pytest.mark.asyncio
async def test_with_chat_direct_methods():
    """Test that with_chat() accepts prompt configuration directly without ChatStepBuilder."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        # Test direct configuration in with_chat()
        agent = await (
            AgentFactory(ChatState)
            .with_chat(prompt_template="Hello", system_prompt="You are helpful")
            .build()
        )

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result

        # Test fluent chaining with convenience methods
        agent2 = await (
            AgentFactory(ChatState)
            .with_chat()
            .with_prompt("Hello")
            .with_system_prompt("You are helpful")
            .build()
        )

        result2 = await agent2.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result2


def test_with_prompt_no_steps_raises_error():
    """Test that with_prompt() raises ValueError when no steps have been added."""
    factory = AgentFactory(ChatState)
    with pytest.raises(
        ValueError, match="No steps have been added to configure prompt for"
    ):
        factory.with_prompt("test prompt")


def test_with_prompt_non_llm_step_raises_error():
    """Test that with_prompt() raises ValueError when the most recent step is not an LLM step."""
    factory = AgentFactory(ChatState)
    factory.add(lambda x: x)  # Add a custom step
    with pytest.raises(
        ValueError, match="The most recent step is not an LLM or React step"
    ):
        factory.with_prompt("test prompt")


def test_with_system_prompt_no_steps_raises_error():
    """Test that with_system_prompt() raises ValueError when no steps have been added."""
    factory = AgentFactory(ChatState)
    with pytest.raises(
        ValueError, match="No steps have been added to configure system prompt for"
    ):
        factory.with_system_prompt("test system prompt")


def test_with_system_prompt_non_llm_step_raises_error():
    """Test that with_system_prompt() raises ValueError when the most recent step is not an LLM step."""
    factory = AgentFactory(ChatState)
    factory.add(lambda x: x)  # Add a custom step
    with pytest.raises(
        ValueError, match="The most recent step is not an LLM or React step"
    ):
        factory.with_system_prompt("test system prompt")


def test_with_structured_output_no_llm_step_raises():
    class MyModel(BaseModel):
        answer: str

    factory = AgentFactory(DefaultState)
    with pytest.raises(ValueError, match="No steps have been added"):
        factory.with_structured_output(MyModel)


def test_with_structured_output_non_llm_step_raises():
    class MyModel(BaseModel):
        answer: str

    async def dummy_step(state):
        return state

    factory = AgentFactory(DefaultState).add(dummy_step)
    with pytest.raises(
        ValueError, match="The most recent step is not an LLM or React step"
    ):
        factory.with_structured_output(MyModel)


@pytest.mark.asyncio
async def test_with_structured_output_returns_model_instance():

    class MyModel(BaseModel):
        answer: str

    class TempDefaultState(TypedDict):
        """Default state schema for agents."""

        messages: Annotated[list, add_messages]
        name: str
        answer: str

    mock_llm = Mock()
    mock_llm.with_structured_output = Mock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(return_value=MyModel(answer="42"))

    with patch(
        "petal.core.steps.llm.LLMStep._create_llm_for_provider", return_value=mock_llm
    ):
        agent = await (
            AgentFactory(TempDefaultState)
            .with_chat(prompt_template="What is the answer?", model="gpt-4o-mini")
            .with_structured_output(MyModel)
            .build()
        )
        result = await agent.arun(
            {
                "messages": [HumanMessage(content="test")],
                "name": "test",
                "answer": "old",
                "blah": {},
            }
        )
        assert isinstance(result, dict)
        assert result["answer"] == "42"


@pytest.mark.asyncio
async def test_with_structured_output_with_key_returns_dict():
    class MyModel(BaseModel):
        answer: str

    class TempDefaultState(TypedDict):
        """Default state schema for agents."""

        messages: Annotated[list, add_messages]
        name: str
        answer: str
        blah: MyModel

    mock_llm = Mock()
    mock_llm.with_structured_output = Mock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(return_value=MyModel(answer="42"))

    with patch(
        "petal.core.steps.llm.LLMStep._create_llm_for_provider", return_value=mock_llm
    ):
        agent = await (
            AgentFactory(TempDefaultState)
            .with_chat(prompt_template="What is the answer?", model="gpt-4o-mini")
            .with_structured_output(MyModel, key="blah")
            .build()
        )
        result = await agent.arun(
            {
                "messages": [HumanMessage(content="test")],
                "name": "test",
                "answer": "old",
                "blah": {},
            }
        )
        assert isinstance(result, dict)
        assert "blah" in result
        assert result["blah"] == {"answer": "42"}
