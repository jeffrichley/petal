from typing import Annotated, Any, Dict, get_type_hints
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from petal.core.agent import Agent
from petal.core.config.yaml import LLMNodeConfig, ReactNodeConfig
from petal.core.factory import AgentFactory, DefaultState
from petal.core.tool_factory import ToolFactory
from petal.core.yaml.parser import YAMLFileNotFoundError, YAMLParseError
from pydantic import BaseModel
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
        agent = (
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
        agent = (
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


@pytest.mark.asyncio
async def test_with_react_tools():
    """Test with_react_tools() with scratchpad support."""
    from langchain_core.tools import tool

    @tool
    def react_tool(query: str) -> str:
        """A tool for ReAct testing."""
        return f"ReAct: {query}"

    agent = (
        AgentFactory(ChatState)
        .with_chat()
        .with_react_tools([react_tool], scratchpad_key="observations")
        .build()
    )

    # Verify the agent was built successfully
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_with_react_tools_default_scratchpad():
    """Test with_react_tools() with default scratchpad key."""
    from langchain_core.tools import tool

    @tool
    def default_tool(query: str) -> str:
        """A tool for default scratchpad testing."""
        return f"Default: {query}"

    agent = (
        AgentFactory(ChatState)
        .with_chat()
        .with_react_tools([default_tool])  # Should use default "scratchpad" key
        .build()
    )

    # Verify the agent was built successfully
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_with_tools_no_scratchpad():
    """Test with_tools() without scratchpad (basic tool support)."""
    from langchain_core.tools import tool

    @tool
    def basic_tool(query: str) -> str:
        """A basic tool without scratchpad."""
        return f"Basic: {query}"

    agent = (
        AgentFactory(ChatState)
        .with_chat()
        .with_tools([basic_tool])  # No scratchpad_key specified
        .build()
    )

    # Verify the agent was built successfully
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_with_tools_custom_scratchpad():
    """Test with_tools() with custom scratchpad key."""
    from langchain_core.tools import tool

    @tool
    def custom_tool(query: str) -> str:
        """A tool with custom scratchpad."""
        return f"Custom: {query}"

    agent = (
        AgentFactory(ChatState)
        .with_chat()
        .with_tools([custom_tool], scratchpad_key="custom_observations")
        .build()
    )

    # Verify the agent was built successfully
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_with_tools_fluent_chaining():
    """Test that with_tools() supports fluent chaining."""
    from langchain_core.tools import tool

    @tool
    def chained_tool(query: str) -> str:
        """A tool for chaining test."""
        return f"Chained: {query}"

    factory = AgentFactory(ChatState)
    result = factory.with_chat().with_tools([chained_tool])

    # Verify fluent chaining returns self
    assert result is factory

    # Verify the agent can still be built
    agent = factory.build()
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_with_react_tools_fluent_chaining():
    """Test that with_react_tools() supports fluent chaining."""
    from langchain_core.tools import tool

    @tool
    def react_chained_tool(query: str) -> str:
        """A tool for ReAct chaining test."""
        return f"ReAct Chained: {query}"

    factory = AgentFactory(ChatState)
    result = factory.with_chat().with_react_tools([react_chained_tool])

    # Verify fluent chaining returns self
    assert result is factory

    # Verify the agent can still be built
    agent = factory.build()
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_tool_factory_integration():
    """Test that AgentFactory properly integrates with ToolFactory."""
    from langchain_core.tools import tool

    @tool
    def factory_tool(query: str) -> str:
        """A tool for factory integration testing."""
        return f"Factory: {query}"

    # Create factory and register tool
    factory = AgentFactory(ChatState)
    factory._tool_factory.add("factory_tool", factory_tool)

    # Use string name - should resolve via ToolFactory
    agent = factory.with_chat().with_tools(["factory_tool"]).build()

    # Verify the agent was built successfully
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_tool_factory_resolution_error():
    """Test that AgentFactory handles ToolFactory resolution errors gracefully."""
    # Try to use a tool name that doesn't exist
    with pytest.raises(KeyError, match="Tool 'nonexistent_tool' not found in registry"):
        (AgentFactory(ChatState).with_chat().with_tools(["nonexistent_tool"]).build())


@pytest.mark.asyncio
async def test_with_tools_empty_list():
    """Test with_tools() with empty tool list raises error."""
    with pytest.raises(
        ValueError, match="Tools list cannot be empty. Provide at least one tool."
    ):
        AgentFactory(ChatState).with_chat().with_tools([]).build()


@pytest.mark.asyncio
async def test_with_react_tools_empty_list():
    """Test with_react_tools() with empty tool list raises error."""
    with pytest.raises(
        ValueError, match="Tools list cannot be empty. Provide at least one tool."
    ):
        AgentFactory(ChatState).with_chat().with_react_tools([]).build()


@pytest.mark.asyncio
async def test_with_tools_before_chat_raises_error():
    """Test that with_tools() before with_chat() raises error."""
    from langchain_core.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """A test tool."""
        return f"Test: {query}"

    with pytest.raises(
        ValueError, match="No steps have been added. Call with_chat\\(\\) first."
    ):
        AgentFactory(ChatState).with_tools([test_tool])


@pytest.mark.asyncio
async def test_with_react_tools_before_chat_raises_error():
    """Test that with_react_tools() before with_chat() raises error."""
    from langchain_core.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """A test tool."""
        return f"Test: {query}"

    with pytest.raises(
        ValueError, match="No steps have been added. Call with_chat\\(\\) first."
    ):
        AgentFactory(ChatState).with_react_tools([test_tool])


@pytest.mark.asyncio
async def test_with_tools_non_llm_step_raises_error():
    """Test that with_tools() when the most recent step is not an LLM step raises error."""
    from langchain_core.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """A test tool."""
        return f"Test: {query}"

    async def custom_step(state):  # noqa: ARG001
        """A custom step that is not an LLM step."""
        return {"custom": "value"}

    # Add a custom step first, then try to add tools
    with pytest.raises(
        ValueError,
        match="The most recent step is not an LLM step. Call with_chat\\(\\) first.",
    ):
        AgentFactory(ChatState).add(custom_step).with_tools([test_tool])


@pytest.mark.asyncio
async def test_with_react_tools_non_llm_step_raises_error():
    """Test that with_react_tools() when the most recent step is not an LLM step raises error."""
    from langchain_core.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """A test tool."""
        return f"Test: {query}"

    async def custom_step(state):  # noqa: ARG001
        """A custom step that is not an LLM step."""
        return {"custom": "value"}

    # Add a custom step first, then try to add tools
    with pytest.raises(
        ValueError,
        match="The most recent step is not an LLM step. Call with_chat\\(\\) first.",
    ):
        AgentFactory(ChatState).add(custom_step).with_react_tools([test_tool])


@pytest.mark.asyncio
async def test_tools_are_injected_and_invoke_tool_message():
    """End-to-end: verify tools are injected, invoked, and ToolMessage is appended."""
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    # --- Tool as direct object ---
    called = {}

    @tool
    async def echo_tool(query: str) -> str:
        """Echoes the input query for testing tool injection."""
        called["direct"] = query
        return f"Echo: {query}"

    # --- Tool as string name ---
    @tool
    async def string_tool(query: str) -> str:
        """Returns a string with the input query for testing tool injection."""
        called["string"] = query
        return f"String: {query}"

    # Minimal mock LLM
    class DummyLLM:
        def __init__(self):
            self._call_count = 0

        async def ainvoke(self, _, config=None, **kwargs):  # noqa: ARG002

            self._call_count += 1
            if self._call_count == 1:
                return AIMessage(
                    content="I need to call a tool",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "string_tool",
                            "args": {"query": "foo"},
                        }
                    ],
                )
            else:
                return AIMessage(content="Tool execution completed successfully")


class TestDiagramAgent:
    """Test the diagram_agent static method."""

    def test_diagram_agent_success_png(self, tmp_path):
        """Test successful diagram generation in PNG format."""
        from langgraph.graph import END, START, StateGraph
        from petal.core.agent import Agent
        from typing_extensions import TypedDict

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

        # Mock the graph object to return a mock with draw_mermaid_png
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_png = Mock(return_value=b"fake_png_data")
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        # Test diagram generation
        output_path = tmp_path / "test_diagram.png"
        AgentFactory.diagram_agent(agent, str(output_path), "png")

        # Verify the mock was called correctly
        mock_graph_obj.draw_mermaid_png.assert_called_once()
        assert output_path.exists()

    def test_diagram_agent_success_svg(self, tmp_path):
        """Test successful diagram generation in SVG format."""
        from langgraph.graph import END, START, StateGraph
        from petal.core.agent import Agent
        from typing_extensions import TypedDict

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

        # Mock the graph object to return a mock with draw_mermaid_svg
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_svg = Mock(return_value=b"fake_svg_data")
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        # Test diagram generation
        output_path = tmp_path / "test_diagram.svg"
        AgentFactory.diagram_agent(agent, str(output_path), "svg")

        # Verify the mock was called correctly
        mock_graph_obj.draw_mermaid_svg.assert_called_once()
        assert output_path.exists()

    def test_diagram_agent_agent_not_built(self):
        """Test that diagram_agent raises error when agent is not built."""
        from petal.core.agent import Agent

        # Create agent without building it
        agent = Agent()
        agent.built = False

        with pytest.raises(
            RuntimeError, match="Agent must be built before generating diagram"
        ):
            AgentFactory.diagram_agent(agent, "test.png")

    def test_diagram_agent_graph_is_none(self):
        """Test that diagram_agent raises error when agent.graph is None."""
        from petal.core.agent import Agent

        # Create agent with built=True but graph=None
        agent = Agent()
        agent.built = True
        agent.graph = None

        with pytest.raises(
            RuntimeError, match="Agent must be built before generating diagram"
        ):
            AgentFactory.diagram_agent(agent, "test.png")

    def test_diagram_agent_unsupported_format(self):
        """Test that diagram_agent raises error for unsupported formats."""
        from langgraph.graph import END, START, StateGraph
        from petal.core.agent import Agent
        from typing_extensions import TypedDict

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

        # Mock the graph object
        mock_graph_obj = Mock()
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        with pytest.raises(RuntimeError, match="Unsupported format: pdf"):
            AgentFactory.diagram_agent(agent, "test.pdf", "pdf")

    def test_diagram_agent_graph_no_mermaid_support(self):
        """Test that diagram_agent raises error when graph doesn't support mermaid."""
        from langgraph.graph import END, START, StateGraph
        from petal.core.agent import Agent
        from typing_extensions import TypedDict

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

        # Create a simple object without mermaid methods
        simple_graph_obj = object()
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(return_value=simple_graph_obj)

        with pytest.raises(
            RuntimeError,
            match="Graph object doesn't support mermaid diagram generation",
        ):
            AgentFactory.diagram_agent(agent, "test.png")

    def test_diagram_agent_file_write_error(self, tmp_path):
        """Test that diagram_agent handles file write errors gracefully."""
        from langgraph.graph import END, START, StateGraph
        from petal.core.agent import Agent
        from typing_extensions import TypedDict

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

        # Mock the graph object
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_png = Mock(return_value=b"fake_png_data")
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        # Test with a path that can't be written to (directory instead of file)
        directory_path = tmp_path / "test_dir"
        directory_path.mkdir()

        with pytest.raises(RuntimeError, match="Failed to generate diagram"):
            AgentFactory.diagram_agent(agent, str(directory_path), "png")

    def test_diagram_agent_graph_get_graph_error(self):
        """Test that diagram_agent handles graph.get_graph() errors gracefully."""
        from langgraph.graph import END, START, StateGraph
        from petal.core.agent import Agent
        from typing_extensions import TypedDict

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

        # Mock get_graph to raise an exception
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(side_effect=Exception("Graph error"))

        with pytest.raises(
            RuntimeError, match="Failed to generate diagram: Graph error"
        ):
            AgentFactory.diagram_agent(agent, "test.png")

    def test_diagram_agent_mermaid_method_error(self):
        """Test that diagram_agent handles mermaid method errors gracefully."""
        from langgraph.graph import END, START, StateGraph
        from petal.core.agent import Agent
        from typing_extensions import TypedDict

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

        # Mock the graph object with a method that raises an exception
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_png = Mock(side_effect=Exception("Mermaid error"))
        # Use setattr to properly mock the method
        assert agent.graph is not None  # type: ignore[unreachable]
        agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        with pytest.raises(
            RuntimeError, match="Failed to generate diagram: Mermaid error"
        ):
            AgentFactory.diagram_agent(agent, "test.png")


class TestDiagramGraph:
    """Test the diagram_graph method."""

    def test_diagram_graph_success(self, tmp_path):
        """Test successful diagram generation with diagram_graph."""
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Test: {query}"

        # Create factory with steps
        factory = AgentFactory(ChatState)
        factory.with_chat().with_tools([test_tool])

        # Mock the build method to return a working agent
        mock_agent = Mock()
        mock_agent.built = True
        mock_agent.graph = Mock()

        # Mock the graph object
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_png = Mock(return_value=b"fake_png_data")
        # Use setattr to properly mock the method
        assert mock_agent.graph is not None  # type: ignore[unreachable]
        mock_agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        factory.build = Mock(return_value=mock_agent)  # type: ignore[method-assign]

        # Test diagram generation
        output_path = tmp_path / "test_diagram.png"
        factory.diagram_graph(str(output_path), "png")

        # Verify the mocks were called correctly
        factory.build.assert_called_once()  # type: ignore[attr-defined]
        mock_graph_obj.draw_mermaid_png.assert_called_once()
        assert output_path.exists()

    def test_diagram_graph_no_steps_raises_error(self):
        """Test that diagram_graph raises error when no steps are configured."""
        factory = AgentFactory(ChatState)
        # Don't add any steps

        with pytest.raises(
            ValueError, match="Cannot generate diagram: no steps have been configured"
        ):
            factory.diagram_graph("test.png")

    def test_diagram_graph_build_failure_propagates(self):
        """Test that diagram_graph propagates build failures."""
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Test: {query}"

        # Create factory with steps
        factory = AgentFactory(ChatState)
        factory.with_chat().with_tools([test_tool])

        # Mock build to raise an exception
        factory.build = Mock(side_effect=Exception("Build failed"))  # type: ignore[method-assign]

        with pytest.raises(Exception, match="Build failed"):
            factory.diagram_graph("test.png")

    def test_diagram_graph_diagram_agent_failure_propagates(self):
        """Test that diagram_graph propagates diagram_agent failures."""
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Test: {query}"

        # Create factory with steps
        factory = AgentFactory(ChatState)
        factory.with_chat().with_tools([test_tool])

        # Mock the build method to return an agent that will fail diagram generation
        mock_agent = Mock()
        mock_agent.built = True
        mock_agent.graph = Mock()

        # Mock get_graph to raise an exception
        # Use setattr to properly mock the method
        assert mock_agent.graph is not None  # type: ignore[unreachable]
        mock_agent.graph.get_graph = Mock(side_effect=Exception("Graph error"))

        factory.build = Mock(return_value=mock_agent)  # type: ignore[method-assign]

        with pytest.raises(
            RuntimeError, match="Failed to generate diagram: Graph error"
        ):
            factory.diagram_graph("test.png")

    def test_diagram_graph_with_different_formats(self, tmp_path):
        """Test diagram_graph with different output formats."""
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Test: {query}"

        # Create factory with steps
        factory = AgentFactory(ChatState)
        factory.with_chat().with_tools([test_tool])

        # Mock the build method to return a working agent
        mock_agent = Mock()
        mock_agent.built = True
        mock_agent.graph = Mock()

        # Mock the graph object for SVG
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_svg = Mock(return_value=b"fake_svg_data")
        # Use setattr to properly mock the method
        assert mock_agent.graph is not None  # type: ignore[unreachable]
        mock_agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        factory.build = Mock(return_value=mock_agent)  # type: ignore[method-assign]

        # Test SVG diagram generation
        output_path = tmp_path / "test_diagram.svg"
        factory.diagram_graph(str(output_path), "svg")

        # Verify the mocks were called correctly
        factory.build.assert_called_once()  # type: ignore[attr-defined]
        mock_graph_obj.draw_mermaid_svg.assert_called_once()
        assert output_path.exists()

    def test_diagram_graph_returns_none(self, tmp_path):
        """Test that diagram_graph returns None (not self for fluent chaining)."""
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Test: {query}"

        # Create factory with steps
        factory = AgentFactory(ChatState)
        factory.with_chat().with_tools([test_tool])

        # Mock the build method to return a working agent
        mock_agent = Mock()
        mock_agent.built = True
        mock_agent.graph = Mock()

        # Mock the graph object
        mock_graph_obj = Mock()
        mock_graph_obj.draw_mermaid_png = Mock(return_value=b"fake_png_data")
        # Use setattr to properly mock the method
        assert mock_agent.graph is not None  # type: ignore[unreachable]
        mock_agent.graph.get_graph = Mock(return_value=mock_graph_obj)

        factory.build = Mock(return_value=mock_agent)  # type: ignore[method-assign]

        # Test that diagram_graph returns None
        output_path = tmp_path / "test_diagram.png"
        result = factory.diagram_graph(str(output_path), "png")

        # Verify it returns None (not self for fluent chaining)
        assert result is None
        assert output_path.exists()


@pytest.mark.asyncio
async def test_with_react_loop():
    """Test with_react_loop() adds a React step with tools."""
    from langchain.tools import tool

    @tool
    def react_tool(query: str) -> str:
        """A test tool for React loop."""
        return f"Processed: {query}"

    tool_factory = ToolFactory()
    tool_factory.add("react_tool", react_tool)

    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        # Create a mock LLM instance with proper async ainvoke method
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Test response")]}
        )
        mock_llm_instance.with_structured_output = Mock(
            return_value=mock_structured_llm
        )

        mock_llm_class.return_value = mock_llm_instance

        factory = AgentFactory(ChatState)
        agent = factory.with_react_loop(
            ["react_tool"], tool_factory=tool_factory
        ).build()
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_react_loop_with_config():
    """Test with_react_loop() with additional configuration."""
    from langchain.tools import tool

    @tool
    def config_tool(query: str) -> str:
        """A test tool for React loop with config."""
        return f"Configured: {query}"

    tool_factory = ToolFactory()
    tool_factory.add("config_tool", config_tool)

    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        # Create a mock LLM instance with proper async ainvoke method
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Test response")]}
        )
        mock_llm_instance.with_structured_output = Mock(
            return_value=mock_structured_llm
        )

        mock_llm_class.return_value = mock_llm_instance

        factory = AgentFactory(ChatState)
        agent = factory.with_react_loop(
            ["config_tool"],
            tool_factory=tool_factory,
            system_prompt="You are a helpful assistant.",
        ).build()
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_react_loop_fluent_chaining():
    """Test that with_react_loop() supports fluent chaining."""
    from langchain.tools import tool

    @tool
    def chained_tool(query: str) -> str:
        """A test tool for fluent chaining."""
        return f"Chained: {query}"

    tool_factory = ToolFactory()
    tool_factory.add("chained_tool", chained_tool)

    factory = AgentFactory(ChatState)
    result = factory.with_react_loop(["chained_tool"], tool_factory=tool_factory)
    assert result is factory


@pytest.mark.asyncio
async def test_with_react_loop_empty_tools_raises_error():
    """Test with_react_loop() with empty tool list raises error."""
    tool_factory = ToolFactory()
    factory = AgentFactory(ChatState)
    with pytest.raises(ValueError, match="React steps require at least one tool"):
        factory.with_react_loop([], tool_factory=tool_factory).build()


@pytest.mark.asyncio
async def test_with_react_loop_with_string_tools():
    """Test with_react_loop() with string tool names."""
    tool_factory = ToolFactory()
    from langchain.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """A test tool for string tool names."""
        return f"String: {query}"

    tool_factory.add("test_tool", test_tool)

    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        # Create a mock LLM instance with proper async ainvoke method
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Test response")]}
        )
        mock_llm_instance.with_structured_output = Mock(
            return_value=mock_structured_llm
        )

        mock_llm_class.return_value = mock_llm_instance

        factory = AgentFactory(ChatState)
        agent = factory.with_react_loop(
            ["test_tool"], tool_factory=tool_factory
        ).build()
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_react_loop_with_structured_output():
    """Test with_react_loop() with structured output model."""
    from langchain.tools import tool
    from pydantic import BaseModel

    class TestOutput(BaseModel):
        answer: str
        confidence: float

    @tool
    def structured_tool(query: str) -> str:
        """A test tool for structured output."""
        return f"Structured: {query}"

    tool_factory = ToolFactory()
    tool_factory.add("structured_tool", structured_tool)

    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        # Create a mock LLM instance with proper async ainvoke method
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Test response")]}
        )
        mock_llm_instance.with_structured_output = Mock(
            return_value=mock_structured_llm
        )

        mock_llm_class.return_value = mock_llm_instance

        factory = AgentFactory(ChatState)
        agent = (
            factory.with_react_loop(["structured_tool"], tool_factory=tool_factory)
            .with_structured_output(TestOutput)
            .build()
        )
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_react_loop_with_prompt_template():
    """Test with_react_loop() with prompt template."""
    from langchain.tools import tool

    @tool
    def prompt_tool(query: str) -> str:
        """A test tool for prompt template."""
        return f"Prompted: {query}"

    tool_factory = ToolFactory()
    tool_factory.add("prompt_tool", prompt_tool)

    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        # Create a mock LLM instance with proper async ainvoke method
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Test response")]}
        )
        mock_llm_instance.with_structured_output = Mock(
            return_value=mock_structured_llm
        )

        mock_llm_class.return_value = mock_llm_instance

        factory = AgentFactory(ChatState)
        agent = (
            factory.with_react_loop(["prompt_tool"], tool_factory=tool_factory)
            .with_prompt("Answer the question: {input}")
            .build()
        )
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_react_loop_with_system_prompt():
    """Test with_react_loop() with system prompt."""
    from langchain.tools import tool

    @tool
    def system_tool(query: str) -> str:
        """A test tool for system prompt."""
        return f"System: {query}"

    tool_factory = ToolFactory()
    tool_factory.add("system_tool", system_tool)

    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        # Create a mock LLM instance with proper async ainvoke method
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Test response")]}
        )
        mock_llm_instance.with_structured_output = Mock(
            return_value=mock_structured_llm
        )

        mock_llm_class.return_value = mock_llm_instance

        factory = AgentFactory(ChatState)
        agent = (
            factory.with_react_loop(["system_tool"], tool_factory=tool_factory)
            .with_system_prompt("You are a helpful assistant.")
            .build()
        )
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


def test_agent_factory_node_from_yaml_llm():
    """Test AgentFactory.node_from_yaml loads LLM node from YAML and adds to builder."""
    factory = AgentFactory(DefaultState)
    mock_config = LLMNodeConfig(
        type="llm",
        name="test_llm",
        description="Test LLM node",
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
        prompt="Test prompt",
        system_prompt="Test system",
    )
    with (
        patch("petal.core.yaml.parser.YAMLNodeParser") as mock_parser,
        patch("petal.core.yaml.handlers.HandlerFactory") as mock_factory,
    ):
        mock_parser.return_value.parse_node_config.return_value = mock_config
        mock_handler = patch("petal.core.yaml.handlers.llm.LLMNodeHandler").start()
        mock_factory.return_value.get_handler.return_value = mock_handler
        mock_handler.create_node.return_value = lambda x: x

        # Should add to builder and return function
        node = factory.node_from_yaml("test.yaml")
        assert callable(node)

        # Should have added a step to the builder
        assert len(factory._builder._config.steps) == 1
        assert factory._builder._config.steps[0].strategy_type == "custom"
        assert factory._builder._config.steps[0].node_name == "test_llm"

        patch.stopall()


def test_agent_factory_node_from_yaml_react():
    """Test AgentFactory.node_from_yaml loads React node from YAML and adds to builder."""
    factory = AgentFactory(DefaultState)
    mock_config = ReactNodeConfig(
        type="react",
        name="test_react",
        description="Test React node",
        tools=["search", "calculator"],
        reasoning_prompt="Think step by step",
        system_prompt="You are a reasoning agent",
        max_iterations=5,
    )
    with (
        patch("petal.core.yaml.parser.YAMLNodeParser") as mock_parser,
        patch("petal.core.yaml.handlers.HandlerFactory") as mock_factory,
    ):
        mock_parser.return_value.parse_node_config.return_value = mock_config
        mock_handler = patch("petal.core.yaml.handlers.react.ReactNodeHandler").start()
        mock_factory.return_value.get_handler.return_value = mock_handler
        mock_handler.create_node.return_value = lambda x: x

        # Should add to builder and return function
        node = factory.node_from_yaml("test.yaml")
        assert callable(node)

        # Should have added a step to the builder
        assert len(factory._builder._config.steps) == 1
        assert factory._builder._config.steps[0].strategy_type == "custom"
        assert factory._builder._config.steps[0].node_name == "test_react"

        patch.stopall()


def test_agent_factory_node_from_yaml_file_not_found():
    """Test AgentFactory.node_from_yaml handles file not found."""
    factory = AgentFactory(DefaultState)
    with pytest.raises(YAMLFileNotFoundError):
        factory.node_from_yaml("nonexistent.yaml")


def test_agent_factory_node_from_yaml_invalid_yaml():
    """Test AgentFactory.node_from_yaml handles invalid YAML."""
    factory = AgentFactory(DefaultState)
    with patch("petal.core.yaml.parser.YAMLNodeParser") as mock_parser:
        mock_parser.return_value.parse_node_config.side_effect = YAMLParseError(
            "Invalid YAML"
        )
        with pytest.raises(YAMLParseError):
            factory.node_from_yaml("invalid.yaml")
