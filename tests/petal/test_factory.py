from typing import Annotated, Any, Dict, get_type_hints
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from petal.core.agent import Agent
from petal.core.factory import AgentFactory, DefaultState
from typing_extensions import TypedDict


# Simple state types for testing
class SimpleState(TypedDict):
    x: int
    processed: bool


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


class MixedState(TypedDict):
    messages: Annotated[list, add_messages]
    processed: bool
    x: int


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_agent_factory_normal():
    async def step1(state):  # noqa: ARG001
        return {"x": 1}

    async def step2(state):  # noqa: ARG001
        # In LangGraph, we need to access the previous step's output
        # Since step1 returns {"x": 1}, this should be available
        return {"x": state.get("x", 0) + 2}

    agent = AgentFactory(SimpleState).add(step1).add(step2).build()
    result = await agent.arun({})
    # Only the last step's fields should be in the result
    assert result["x"] == 3


@pytest.mark.asyncio
async def test_agent_factory_no_steps():
    factory = AgentFactory(SimpleState)
    with pytest.raises(ValueError, match="Cannot build agent: no steps configured"):
        factory.build()


@pytest.mark.asyncio
async def test_agent_arun_before_build():
    """Test that agent arun fails when not built."""
    # Create an agent without building it
    agent = Agent()  # Create agent with no graph

    with pytest.raises(RuntimeError):
        await agent.arun({})


@pytest.mark.asyncio
async def test_agent_arun_with_none_graph():
    """Test that agent arun fails when graph is None but agent thinks it's built."""
    agent = Agent()
    # Manually set built=True but leave graph=None to test the specific error path
    agent.built = True
    agent.graph = None

    with pytest.raises(
        RuntimeError, match="Agent.graph is None - agent was not properly built"
    ):
        await agent.arun({})


@pytest.mark.asyncio
async def test_agent_build_method():
    """Test that agent build method works correctly."""
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    # Create a proper state type for the test
    class TestState(TypedDict):
        test: str

    # Create a simple graph
    graph = StateGraph(TestState)
    graph.add_node("test", lambda x: x)
    graph.add_edge(START, "test")
    graph.add_edge("test", END)
    compiled_graph = graph.compile()

    # Create and build agent
    agent = Agent().build(compiled_graph, TestState)

    # Test that agent is built
    assert agent.built is True
    assert agent.graph is compiled_graph
    assert agent.state_type is TestState

    # Test that agent can run
    result = await agent.arun({"test": "value"})
    assert result == {"test": "value"}


@pytest.mark.asyncio
async def test_with_chat_default_openai():
    """Test that with_chat() with None uses default OpenAI config."""
    from unittest.mock import patch

    from langchain_core.messages import AIMessage

    # Create a mock LLM instance
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))

    # Mock the provider-specific LLM creation method
    with patch(
        "petal.core.steps.llm.LLMStep._create_llm_for_provider", return_value=mock_llm
    ):
        agent = AgentFactory(ChatState).with_chat().add(lambda s: s).build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_chat_custom_config():
    """Test that with_chat() accepts a custom config dict."""
    from unittest.mock import patch

    from langchain_core.messages import AIMessage

    # Create a mock LLM instance
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))

    config = {"provider": "openai", "model": "gpt-4o", "temperature": 0.2}

    # Mock the provider-specific LLM creation method
    with patch(
        "petal.core.steps.llm.LLMStep._create_llm_for_provider", return_value=mock_llm
    ):
        agent = (
            AgentFactory(ChatState)
            .with_chat(llm_config=config)
            .add(lambda s: s)
            .build()
        )

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_chat_custom_llm_instance():
    """Test that with_chat() accepts a pre-configured LLM instance."""
    from langchain.chat_models.base import BaseChatModel

    # Create a mock that inherits from BaseChatModel
    mock_llm = Mock(spec=BaseChatModel)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))

    agent = AgentFactory(ChatState).with_chat(mock_llm).add(lambda s: s).build()

    result = await agent.arun({"messages": [HumanMessage(content="test")]})
    assert "messages" in result


@pytest.mark.asyncio
async def test_with_chat_kwargs():
    """Test that with_chat() accepts kwargs for configuration."""
    from unittest.mock import patch

    from langchain_core.messages import AIMessage

    # Create a mock LLM instance
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))

    # Mock the provider-specific LLM creation method
    with patch(
        "petal.core.steps.llm.LLMStep._create_llm_for_provider", return_value=mock_llm
    ):
        agent = (
            AgentFactory(ChatState)
            .with_chat(model="gpt-4o", temperature=0.1)
            .add(lambda s: s)
            .build()
        )

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_chat_fluent_interface():
    """Test that with_chat() returns ChatStepBuilder for chaining."""
    factory = AgentFactory(ChatState)
    result = factory.with_chat()

    # Test that the result supports fluent chaining
    result2 = result.with_prompt("Test prompt")
    result3 = result2.with_system_prompt("Test system")
    assert result3 is not None


@pytest.mark.asyncio
async def test_agent_with_llm():
    """Test that Agent can be built with an LLM."""

    async def step(state):  # noqa: ARG001
        return {"processed": True}

    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(MixedState).add(step).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 2  # Original + AI response


@pytest.mark.asyncio
async def test_agent_without_llm():
    """Test that Agent can be built without an LLM."""

    async def step(state):  # noqa: ARG001
        return {"processed": True}

    agent = AgentFactory(SimpleState).add(step).build()

    result = await agent.arun({})
    # Only the last step's fields should be in the result
    assert result["processed"] is True


@pytest.mark.asyncio
async def test_with_chat_multiple_calls():
    """Test that multiple with_chat() calls create separate LLM steps."""
    from langchain.chat_models.base import BaseChatModel

    mock_llm1 = Mock(spec=BaseChatModel)
    mock_llm2 = Mock(spec=BaseChatModel)
    mock_llm1.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
    mock_llm2.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))

    agent = AgentFactory(ChatState).with_chat(mock_llm1).with_chat(mock_llm2).build()

    result = await agent.arun({"messages": [HumanMessage(content="test")]})
    assert "messages" in result


@pytest.mark.asyncio
async def test_with_chat_invalid_input():
    """Test that with_chat() raises ValueError for invalid input types."""
    factory = AgentFactory(ChatState)
    with pytest.raises(ValueError, match="llm must be a BaseChatModel"):
        factory.with_chat("invalid")


@pytest.mark.asyncio
async def test_with_chat_integration():
    """Test full integration of with_chat() in a complete agent build."""

    async def step1(state):  # noqa: ARG001
        return {"step": 1}

    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(MixedState).add(step1).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 2  # Original + AI response


@pytest.mark.asyncio
async def test_chat_step_builder_with_prompt():
    """Test that ChatStepBuilder can configure prompts."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = (
            AgentFactory(ChatState)
            .with_chat()
            .with_prompt("Test prompt")
            .with_system_prompt("You are a helpful assistant")
            .add(lambda _: {"step": "value"})
            .build()
        )

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 3  # Original + user prompt + AI response


@pytest.mark.asyncio
async def test_chat_step_builder_fluent_interface():
    """Test that ChatStepBuilder supports fluent chaining."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        builder = AgentFactory(ChatState).with_chat()
        result1 = builder.with_prompt("Test")
        result2 = result1.with_system_prompt("System")
        result3 = result2.add(lambda s: s)
        agent = result3.build()

        assert agent is not None
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_unsupported_provider():
    """Test that unsupported LLM providers raise an error."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_chat_openai.side_effect = Exception("Unsupported provider")

        agent = (
            AgentFactory(ChatState)
            .with_chat(llm_config={"provider": "unsupported"})
            .add(lambda s: s)
            .build()
        )

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            await agent.arun({"messages": [HumanMessage(content="test")]})


@pytest.mark.asyncio
async def test_agent_run_with_llm_auto_invoke():
    """Test that LLM is automatically invoked during agent execution."""

    async def step(state):  # noqa: ARG001
        return {"processed": True}

    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(MixedState).add(step).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 2  # Original + AI response
        mock_llm.ainvoke.assert_called()


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_agent_arun_with_llm_auto_invoke():
    """Test that LLM is automatically invoked during async agent execution."""

    async def step(state):  # noqa: ARG001
        return {"processed": True}

    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(MixedState).add(step).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_agent_run_without_llm():
    """Test that agent works without LLM steps."""

    async def step(state):  # noqa: ARG001
        return {"processed": True}

    agent = AgentFactory(SimpleState).add(step).build()

    result = await agent.arun({})
    # Only the last step's fields should be in the result
    assert result["processed"] is True


@pytest.mark.asyncio
async def test_agent_with_custom_node_names():
    """Test that custom node names can be provided."""

    async def step1(state):  # noqa: ARG001
        return {"x": 1}

    async def step2(state):  # noqa: ARG001
        # In LangGraph, we need to access the previous step's output
        return {"x": state.get("x", 0) + 2}

    agent = (
        AgentFactory(SimpleState)
        .add(step1, "custom_step_1")
        .add(step2, "custom_step_2")
        .build()
    )
    result = await agent.arun({})
    # Only the last step's fields should be in the result
    assert result["x"] == 3


@pytest.mark.asyncio
async def test_agent_with_typed_dict_state():
    """Test that TypedDict state types work correctly."""
    from typing_extensions import TypedDict

    class MyState(TypedDict):
        messages: list
        processed: bool

    async def step(_state: MyState) -> Dict[str, Any]:
        return {"processed": True}

    agent = AgentFactory(state_type=MyState).add(step).build()

    result = await agent.arun({"messages": [], "processed": False})
    assert result["processed"] is True
    assert "messages" in result


@pytest.mark.asyncio
async def test_agent_with_prompt_template():
    """Test that prompt templates work correctly."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = (
            AgentFactory(ChatState)
            .with_chat()
            .with_prompt("Process with step")
            .with_system_prompt("You are a helpful assistant")
            .build()
        )

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 3  # Original + user prompt + AI response


@pytest.mark.asyncio
async def test_agent_with_system_prompt_only():
    """Test that system prompt works without user prompt template."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = (
            AgentFactory(ChatState)
            .with_chat()
            .with_system_prompt("You are a helpful assistant")
            .build()
        )

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_agent_with_no_prompts():
    """Test that LLM step works without any prompts."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_agent_with_typed_dict_return():
    """Test that TypedDict states return updates correctly."""
    from typing_extensions import TypedDict

    class MyState(TypedDict):
        messages: list
        processed: bool

    async def step(_state: MyState) -> Dict[str, Any]:
        return {"processed": True}

    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(state_type=MyState).add(step).with_chat().build()

        result = await agent.arun({"messages": [], "processed": False})
        assert result["processed"] is True
        assert "messages" in result


@pytest.mark.asyncio
async def test_agent_with_empty_messages():
    """Test that agent handles empty messages correctly."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().build()

        result = await agent.arun({"messages": []})
        assert "messages" in result


@pytest.mark.asyncio
async def test_agent_with_no_messages_key():
    """Test that agent handles missing messages key correctly."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().build()

        result = await agent.arun({})
        assert "messages" in result


@pytest.mark.asyncio
async def test_agent_with_async_dispatch():
    """Test that async dispatch works correctly."""

    async def step(state):  # noqa: ARG001
        return {"processed": True}

    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(MixedState).add(step).with_chat().build()

        # Create a state with __async__ flag
        state = {"messages": [HumanMessage(content="test")], "__async__": True}
        result = await agent.arun(state)
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 2  # Original + AI response


@pytest.mark.asyncio
async def test_agent_with_custom_state_type_with_messages():
    """Test that custom state types with existing messages field work correctly."""
    from typing_extensions import TypedDict

    class CustomStateWithMessages(TypedDict):
        messages: Annotated[list, add_messages]
        custom_field: str

    async def step(state):  # noqa: ARG001
        return {"custom_field": "updated"}

    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(CustomStateWithMessages).add(step).with_chat().build()

        result = await agent.arun(
            {"messages": [HumanMessage(content="test")], "custom_field": "initial"}
        )
        assert "messages" in result
        assert "custom_field" in result


@pytest.mark.asyncio
async def test_agent_with_custom_state_type_without_messages():
    """Test that custom state types without messages field get messages added."""
    from typing_extensions import TypedDict

    class CustomStateWithoutMessages(TypedDict):
        custom_field: str

    async def step(state):  # noqa: ARG001
        return {"custom_field": "updated"}

    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(CustomStateWithoutMessages).add(step).with_chat().build()

        result = await agent.arun({"custom_field": "initial"})
        # The custom state type should have messages added automatically when using chat
        assert "messages" in result
        assert "custom_field" in result


@pytest.mark.asyncio
async def test_agent_with_none_state_type_and_chat():
    """Test that None state type with chat raises an error since state_type is required."""

    async def step(state):  # noqa: ARG001
        return {"custom_field": "updated"}

    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        # Since state_type is now required, passing None should raise an error
        with pytest.raises(TypeError):
            AgentFactory(None).add(step).with_chat().build()  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_agent_with_none_state_type_without_chat():
    """Test that None state type without chat raises an error since state_type is required."""

    async def step(state):  # noqa: ARG001
        return {"custom_field": "updated"}

    # Since state_type is now required, passing None should raise an error
    with pytest.raises(TypeError):
        AgentFactory(None).add(step).build()  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_llm_step_with_prompt_template_formatting():
    """Test that LLM step properly formats prompt templates with state variables."""
    from typing_extensions import TypedDict

    class PromptState(TypedDict):
        messages: list
        input_data: str
        step_count: int

    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = (
            AgentFactory(PromptState)
            .with_chat()
            .with_prompt("Process {input_data} with {step_count} steps")
            .build()
        )

        # Provide all required keys for prompt formatting
        state = {
            "messages": [HumanMessage(content="test")],
            "input_data": "test_data",
            "step_count": 5,
        }
        try:
            result = await agent.arun(state)
            assert "messages" in result
        except KeyError as e:
            raise AssertionError(
                f"Prompt formatting failed due to missing key: {e}"
            ) from e


@pytest.mark.asyncio
async def test_llm_step_with_system_prompt_only():
    """Test that LLM step works with only system prompt."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = (
            AgentFactory(ChatState)
            .with_chat()
            .with_system_prompt("You are a helpful assistant")
            .build()
        )

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_no_prompts():
    """Test that LLM step works without any prompts."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_empty_messages():
    """Test that LLM step handles empty messages correctly."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().build()

        result = await agent.arun({"messages": []})
        assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_missing_messages_key():
    """Test that LLM step handles missing messages key correctly."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().build()

        result = await agent.arun({})
        assert "messages" in result


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_llm_step_async():
    """Test that async LLM step works correctly."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        # Return a real AIMessage for both sync and async
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().build()
        state = {"messages": [HumanMessage(content="test")], "__async__": True}
        result = await agent.arun(state)
        assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_custom_llm_instance():
    """Test that LLM step works with custom LLM instance."""
    from langchain.chat_models.base import BaseChatModel

    mock_llm = Mock(spec=BaseChatModel)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))

    agent = AgentFactory(ChatState).with_chat(mock_llm).build()

    result = await agent.arun({"messages": [HumanMessage(content="test")]})
    assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_custom_config():
    """Test that LLM step works with custom config."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        config = {"provider": "openai", "model": "gpt-4o", "temperature": 0.2}
        agent = AgentFactory(ChatState).with_chat(llm_config=config).build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        mock_chat_openai.assert_called_once_with(model="gpt-4o", temperature=0.2)
        assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_default_config():
    """Test that LLM step uses default config when none provided."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        mock_chat_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0)
        assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_unsupported_provider():
    """Test that LLM step raises error for unsupported provider."""
    config = {"provider": "unsupported", "model": "test"}
    agent = AgentFactory(ChatState).with_chat(llm_config=config).build()
    # The error is raised at runtime, not build time
    with pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"):
        await agent.arun({"messages": [HumanMessage(content="test")]})


@pytest.mark.asyncio
async def test_agent_build_with_multiple_llm_steps():
    """Test that building agent with multiple LLM steps works correctly."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(ChatState).with_chat().with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_agent_build_with_mixed_steps():
    """Test that building agent with mixed regular and LLM steps works correctly."""

    async def regular_step(state):  # noqa: ARG001
        return {"processed": True}

    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = AgentFactory(MixedState).add(regular_step).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_agent_build_with_custom_node_names():
    """Test that building agent with custom node names works correctly."""

    async def step1(state):  # noqa: ARG001
        return {"x": 1}

    async def step2(state):  # noqa: ARG001
        return {"x": state.get("x", 0) + 2}

    agent = (
        AgentFactory(SimpleState)
        .add(step1, "custom_step1")
        .add(step2, "custom_step2")
        .build()
    )

    result = await agent.arun({})
    assert result["x"] == 3


@pytest.mark.asyncio
async def test_agent_build_with_single_step():
    """Test that building agent with single step works correctly."""

    async def step(state):  # noqa: ARG001
        return {"processed": True}

    agent = AgentFactory(SimpleState).add(step).build()

    result = await agent.arun({})
    assert result["processed"] is True


@pytest.mark.asyncio
async def test_agent_build_with_no_steps_raises():
    """Test that building agent with no steps raises error."""
    factory = AgentFactory(SimpleState)

    with pytest.raises(ValueError, match="Cannot build agent: no steps configured"):
        factory.build()


@pytest.mark.asyncio
async def test_agent_arun_with_built_flag():
    """Test that agent arun works when properly built."""
    # Create a simple agent that can be built
    agent = AgentFactory(SimpleState).add(lambda _: {"processed": True}).build()

    # Test that the agent can run
    result = await agent.arun({})
    assert result["processed"] is True


@pytest.mark.asyncio
async def test_llm_step_prompt_template_missing_key():
    """Test that a helpful ValueError is raised if the prompt template references a missing key."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = (
            AgentFactory(ChatState)
            .with_chat()
            .with_prompt("Hello {missing_key}")
            .build()
        )
        # The state does not include 'missing_key'
        with pytest.raises(ValueError) as exc_info:
            await agent.arun({"messages": [HumanMessage(content="hi")]})
        assert "missing_key" in str(exc_info.value)
        assert "available in the state" in str(exc_info.value)
        assert "messages" in str(exc_info.value)  # should list available keys


@pytest.mark.asyncio
async def test_agent_factory_uses_new_architecture_internally():
    """Test that AgentFactory uses new architecture internally while maintaining backward compatibility."""
    from petal.core.builders.agent import AgentBuilder

    # Test that AgentFactory uses AgentBuilder internally
    factory = AgentFactory(SimpleState)

    # Verify that the internal builder is an AgentBuilder
    assert hasattr(factory, "_builder")
    assert isinstance(factory._builder, AgentBuilder)

    # Verify that the builder has the correct state type
    assert factory._builder._config.state_type == SimpleState

    # Test that add() method uses new architecture
    async def test_step(state):  # noqa: ARG001
        return {"x": 1}

    factory.add(test_step)

    # Verify that the step was added to the new architecture
    assert len(factory._builder._config.steps) == 1
    assert factory._builder._config.steps[0].strategy_type == "custom"
    assert factory._builder._config.steps[0].config["step_function"] == test_step

    # Test that build() uses new architecture
    agent = factory.build()
    assert agent is not None
    assert hasattr(agent, "arun")


class TestLLMStepSyncAsyncHandling:
    """Test that LLM steps work correctly in both sync and async contexts."""

    @pytest.mark.asyncio
    async def test_llm_step_sync_context(self):
        """Test that LLM step works correctly in sync context."""
        factory = AgentFactory(DefaultState)

        from langchain.chat_models.base import BaseChatModel
        from langchain_core.messages import AIMessage, HumanMessage

        mock_llm = Mock(spec=BaseChatModel)
        mock_response = AIMessage(content="Test response")
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = factory.with_chat(llm=mock_llm).with_prompt("Hello {name}").build()

        # Test in async context
        result = await agent.arun(
            {"messages": [HumanMessage(content="hi")], "name": "World"}
        )

        mock_llm.ainvoke.assert_called_once()
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # system prompt + user prompt
        assert call_args[1]["content"] == "Hello World"
        assert "messages" in result
        assert (
            len(result["messages"]) == 3
        )  # original message + user prompt + AI response

    @pytest.mark.asyncio
    async def test_llm_step_async_context(self):
        """Test that LLM step works correctly in async context."""
        factory = AgentFactory(DefaultState)

        from langchain.chat_models.base import BaseChatModel
        from langchain_core.messages import AIMessage, HumanMessage

        mock_llm = Mock(spec=BaseChatModel)
        mock_response = AIMessage(content="Test response")
        # Set up both sync and async methods
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = factory.with_chat(llm=mock_llm).with_prompt("Hello {name}").build()

        result = await agent.arun(
            {"messages": [HumanMessage(content="hi")], "name": "World"}
        )
        mock_llm.ainvoke.assert_called_once()
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # system prompt + user prompt
        assert call_args[1]["content"] == "Hello World"
        assert "messages" in result
        assert (
            len(result["messages"]) == 3
        )  # original message + user prompt + AI response

    @pytest.mark.asyncio
    async def test_llm_step_with_system_prompt(self):
        """Test LLM step with system prompt."""
        factory = AgentFactory(DefaultState)

        from langchain.chat_models.base import BaseChatModel
        from langchain_core.messages import AIMessage, HumanMessage

        mock_llm = Mock(spec=BaseChatModel)
        mock_response = AIMessage(content="Test response")
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = (
            factory.with_chat(llm=mock_llm)
            .with_system_prompt("You are a helpful assistant.")
            .with_prompt("Hello {name}")
            .build()
        )
        await agent.arun({"messages": [HumanMessage(content="hi")], "name": "World"})
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 3  # system prompt + messages + user prompt
        assert call_args[0]["role"] == "system"
        assert call_args[0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_llm_step_without_prompt(self):
        """Test LLM step without user prompt (only system prompt)."""
        factory = AgentFactory(DefaultState)

        from langchain.chat_models.base import BaseChatModel
        from langchain_core.messages import AIMessage, HumanMessage

        mock_llm = Mock(spec=BaseChatModel)
        mock_response = AIMessage(content="Test response")
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = (
            factory.with_chat(llm=mock_llm)
            .with_system_prompt("You are a helpful assistant.")
            .build()
        )
        await agent.arun({"messages": [HumanMessage(content="Hello")]})
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # system prompt + existing message
        assert call_args[0]["role"] == "system"
        assert getattr(call_args[1], "content", None) == "Hello"

    @pytest.mark.asyncio
    async def test_llm_step_with_existing_messages(self):
        """Test LLM step with existing messages in state."""
        factory = AgentFactory(DefaultState)

        from langchain.chat_models.base import BaseChatModel
        from langchain_core.messages import AIMessage, HumanMessage

        mock_llm = Mock(spec=BaseChatModel)
        mock_response = AIMessage(content="Test response")
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = factory.with_chat(llm=mock_llm).with_prompt("Continue").build()
        existing_messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        await agent.arun({"messages": existing_messages, "name": "World"})
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 3  # existing messages + user prompt
        assert getattr(call_args[0], "content", None) == "Hello"
        assert getattr(call_args[1], "content", None) == "Hi there!"
        assert call_args[2]["content"] == "Continue"


@pytest.mark.asyncio
async def test_dynamic_state_type_cache():
    """Test that the dynamic state type cache works properly."""
    from typing_extensions import TypedDict

    class CustomState(TypedDict):
        custom_field: str

    # Create two agents with the same state type and chat
    agent1 = AgentFactory(CustomState).with_chat().build()
    agent2 = AgentFactory(CustomState).with_chat().build()

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

        agent3 = AgentFactory(CustomState).add(step).with_chat().build()
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
        agent = (
            AgentFactory(ChatState)
            .with_chat(prompt_template="Hello", system_prompt="You are helpful")
            .build()
        )

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result

        # Test fluent chaining with convenience methods
        agent2 = (
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
    with pytest.raises(ValueError, match="The most recent step is not an LLM step"):
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
    with pytest.raises(ValueError, match="The most recent step is not an LLM step"):
        factory.with_system_prompt("test system prompt")
