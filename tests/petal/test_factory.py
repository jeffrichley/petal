from typing import Annotated
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from petal.core.factory import Agent, AgentFactory
from typing_extensions import TypedDict


# Simple state types for testing
class TestState(TypedDict):
    x: int
    processed: bool


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


class MixedState(TypedDict):
    messages: Annotated[list, add_messages]
    processed: bool
    x: int


def test_agent_factory_init() -> None:
    af = AgentFactory(TestState)
    assert isinstance(af, AgentFactory)


def test_agent_factory_normal():
    def step1(state):
        return {"x": 1}

    def step2(state):
        # In LangGraph, we need to access the previous step's output
        # Since step1 returns {"x": 1}, this should be available
        return {"x": state.get("x", 0) + 2}

    agent = AgentFactory(TestState).add(step1).add(step2).build()
    result = agent.run({})
    # Only the last step's fields should be in the result
    assert result["x"] == 3


def test_agent_factory_no_steps():
    factory = AgentFactory(TestState)
    with pytest.raises(RuntimeError):
        factory.build()


def test_agent_run_before_build():
    agent = Agent(None)
    agent.built = False
    with pytest.raises(RuntimeError):
        agent.run({})


def test_agent_arun_before_build():
    agent = Agent(None)
    agent.built = False
    with pytest.raises(RuntimeError):
        import asyncio

        asyncio.run(agent.arun({}))


def test_with_chat_default_openai():
    """Test that with_chat() with None uses default OpenAI config."""
    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().add(lambda s: s).build()

        result = agent.run({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


def test_with_chat_custom_config():
    """Test that with_chat() accepts a custom config dict."""
    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        config = {"provider": "openai", "model": "gpt-4o", "temperature": 0.2}
        agent = AgentFactory(ChatState).with_chat(config).add(lambda s: s).build()

        result = agent.run({"messages": [HumanMessage(content="test")]})
        mock_chat_openai.assert_called_once_with(model="gpt-4o", temperature=0.2)
        assert "messages" in result


def test_with_chat_custom_llm_instance():
    """Test that with_chat() accepts a pre-configured LLM instance."""
    from langchain.chat_models.base import BaseChatModel

    # Create a mock that inherits from BaseChatModel
    mock_llm = Mock(spec=BaseChatModel)
    mock_llm.invoke.return_value = AIMessage(content="Test response")

    agent = AgentFactory(ChatState).with_chat(mock_llm).add(lambda s: s).build()

    result = agent.run({"messages": [HumanMessage(content="test")]})
    assert "messages" in result


def test_with_chat_kwargs():
    """Test that with_chat() accepts kwargs for configuration."""
    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = (
            AgentFactory(ChatState)
            .with_chat(model="gpt-4o", temperature=0.1)
            .add(lambda s: s)
            .build()
        )

        result = agent.run({"messages": [HumanMessage(content="test")]})
        mock_chat_openai.assert_called_once_with(model="gpt-4o", temperature=0.1)
        assert "messages" in result


def test_with_chat_fluent_interface():
    """Test that with_chat() returns ChatStepBuilder for chaining."""
    factory = AgentFactory(ChatState)
    result = factory.with_chat()

    assert hasattr(result, "with_prompt")
    assert hasattr(result, "with_system_prompt")


def test_agent_with_llm():
    """Test that Agent can be built with an LLM."""

    def step(state):
        return {"processed": True}

    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(MixedState).add(step).with_chat().build()

        result = agent.run({"messages": [HumanMessage(content="test")]})
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 2  # Original + AI response


def test_agent_without_llm():
    """Test that Agent can be built without an LLM."""

    def step(state):
        return {"processed": True}

    agent = AgentFactory(TestState).add(step).build()

    result = agent.run({})
    # Only the last step's fields should be in the result
    assert result["processed"] is True


def test_with_chat_multiple_calls():
    """Test that multiple with_chat() calls create separate LLM steps."""
    from langchain.chat_models.base import BaseChatModel

    mock_llm1 = Mock(spec=BaseChatModel)
    mock_llm2 = Mock(spec=BaseChatModel)
    mock_llm1.invoke.return_value = AIMessage(content="Test response")
    mock_llm2.invoke.return_value = AIMessage(content="Test response")

    agent = AgentFactory(ChatState).with_chat(mock_llm1).with_chat(mock_llm2).build()

    result = agent.run({"messages": [HumanMessage(content="test")]})
    assert "messages" in result


def test_with_chat_invalid_input():
    """Test that with_chat() raises error for invalid input."""
    factory = AgentFactory(ChatState)

    with pytest.raises(
        ValueError, match="llm must be None, a BaseChatModel instance, or a dict"
    ):
        factory.with_chat("invalid")


def test_with_chat_integration():
    """Test full integration of with_chat() in a complete agent build."""

    def step1(state):
        return {"step": 1}

    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(MixedState).add(step1).with_chat().build()

        result = agent.run({"messages": [HumanMessage(content="test")]})
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 2  # Original + AI response


def test_chat_step_builder_with_prompt():
    """Test that ChatStepBuilder can configure prompts."""
    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = (
            AgentFactory(ChatState)
            .with_chat()
            .with_prompt("Test prompt")
            .with_system_prompt("You are a helpful assistant")
            .add(lambda s: {"step": "value"})
            .build()
        )

        result = agent.run({"messages": [HumanMessage(content="test")]})
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 3  # Original + user prompt + AI response


def test_chat_step_builder_fluent_interface():
    """Test that ChatStepBuilder supports fluent chaining."""
    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        builder = AgentFactory(ChatState).with_chat()
        result1 = builder.with_prompt("Test")
        result2 = result1.with_system_prompt("System")
        result3 = result2.add(lambda s: s)
        agent = result3.build()

        assert agent is not None
        result = agent.run({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


def test_unsupported_provider():
    """Test that unsupported LLM providers raise an error."""
    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_chat_openai.side_effect = Exception("Unsupported provider")

        agent = (
            AgentFactory(ChatState)
            .with_chat({"provider": "unsupported"})
            .add(lambda s: s)
            .build()
        )

        with pytest.raises(Exception):
            agent.run({"messages": [HumanMessage(content="test")]})


def test_agent_run_with_llm_auto_invoke():
    """Test that LLM is automatically invoked during agent execution."""

    def step(state):
        return {"processed": True}

    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(MixedState).add(step).with_chat().build()

        result = agent.run({"messages": [HumanMessage(content="test")]})
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 2  # Original + AI response
        mock_llm.invoke.assert_called()


@pytest.mark.asyncio
async def test_agent_arun_with_llm_auto_invoke():
    """Test that LLM is automatically invoked during async agent execution."""

    def step(state):
        return {"processed": True}

    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        # Set both sync and async methods to return real AIMessage
        mock_llm.invoke.return_value = AIMessage(content="Test response")

        async def async_invoke(messages):
            return AIMessage(content="Test response")

        mock_llm.ainvoke = async_invoke
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(MixedState).add(step).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 2  # Original + AI response


def test_agent_run_without_llm():
    """Test that agent works without LLM steps."""

    def step(state):
        return {"processed": True}

    agent = AgentFactory(TestState).add(step).build()

    result = agent.run({})
    # Only the last step's fields should be in the result
    assert result["processed"] is True


def test_agent_with_custom_node_names():
    """Test that custom node names can be provided."""

    def step1(state):
        return {"x": 1}

    def step2(state):
        # In LangGraph, we need to access the previous step's output
        return {"x": state.get("x", 0) + 2}

    agent = (
        AgentFactory(TestState)
        .add(step1, "custom_step_1")
        .add(step2, "custom_step_2")
        .build()
    )
    result = agent.run({})
    # Only the last step's fields should be in the result
    assert result["x"] == 3


def test_agent_with_typed_dict_state():
    """Test that TypedDict state types work correctly."""
    from typing_extensions import TypedDict

    class MyState(TypedDict):
        messages: list
        processed: bool

    def step(state: MyState) -> MyState:
        return {"processed": True}

    agent = AgentFactory(state_type=MyState).add(step).build()

    result = agent.run({"messages": [], "processed": False})
    assert result["processed"] is True
    assert "messages" in result


def test_agent_with_prompt_template():
    """Test that prompt templates work correctly."""
    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = (
            AgentFactory(ChatState)
            .with_chat()
            .with_prompt("Process with step")
            .with_system_prompt("You are a helpful assistant")
            .build()
        )

        result = agent.run({"messages": [HumanMessage(content="test")]})
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 3  # Original + user prompt + AI response


def test_agent_with_system_prompt_only():
    """Test that system prompt works without user prompt template."""
    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = (
            AgentFactory(ChatState)
            .with_chat()
            .with_system_prompt("You are a helpful assistant")
            .build()
        )

        result = agent.run({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


def test_agent_with_no_prompts():
    """Test that LLM step works without any prompts."""
    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().build()

        result = agent.run({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


def test_agent_with_memory():
    """Test that memory support works."""
    mock_memory = Mock()
    mock_memory.load_memory_variables.return_value = {"memory_key": "memory_value"}

    def step(state):
        return {"processed": True}

    agent = AgentFactory(TestState).add(step).with_memory(mock_memory).build()

    result = agent.run({})
    # Only the last step's fields should be in the result
    assert result["processed"] is True


def test_agent_with_memory_kwargs():
    """Test that memory with kwargs works."""

    def step(state):
        return {"processed": True}

    agent = (
        AgentFactory(TestState)
        .add(step)
        .with_memory(memory_type="test", max_tokens=100)
        .build()
    )

    result = agent.run({})
    # Only the last step's fields should be in the result
    assert result["processed"] is True


def test_agent_with_typed_dict_return():
    """Test that TypedDict states return updates correctly."""
    from typing_extensions import TypedDict

    class MyState(TypedDict):
        messages: list
        processed: bool

    def step(state: MyState) -> MyState:
        return {"processed": True}

    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(state_type=MyState).add(step).with_chat().build()

        result = agent.run({"messages": [], "processed": False})
        assert result["processed"] is True
        assert "messages" in result


def test_agent_with_empty_messages():
    """Test that agent handles empty messages correctly."""
    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().build()

        result = agent.run({"messages": []})
        assert "messages" in result


def test_agent_with_no_messages_key():
    """Test that agent handles missing messages key correctly."""
    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().build()

        result = agent.run({})
        assert "messages" in result


def test_agent_with_async_dispatch():
    """Test that async dispatch works correctly."""

    def step(state):
        return {"processed": True}

    with patch("petal.core.factory.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(MixedState).add(step).with_chat().build()

        # Create a state with __async__ flag
        state = {"messages": [HumanMessage(content="test")], "__async__": True}
        result = agent.run(state)
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 2  # Original + AI response


def test_chat_step_builder_methods():
    """Test all ChatStepBuilder methods."""
    builder = AgentFactory(ChatState).with_chat()

    # Test with_prompt
    result1 = builder.with_prompt("Test prompt")
    assert result1 is builder

    # Test with_system_prompt
    result2 = builder.with_system_prompt("Test system")
    assert result2 is builder

    # Test add
    result3 = builder.add(lambda s: s)
    assert isinstance(result3, AgentFactory)

    # Test with_chat
    result4 = builder.with_chat()
    assert hasattr(result4, "with_prompt")

    # Test build
    agent = builder.build()
    assert isinstance(agent, Agent)
