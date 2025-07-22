from typing import Annotated, Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from tests.petal.conftest_factory import (
    ChatState,
    MixedState,
    SimpleState,
    add_messages,
)

from petal.core.factory import AgentFactory


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
        agent = await AgentFactory(ChatState).with_chat().add(lambda s: s).build()
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
        agent = await (
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
    agent = await AgentFactory(ChatState).with_chat(mock_llm).add(lambda s: s).build()
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
        agent = await (
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

        agent = await AgentFactory(MixedState).add(step).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        # Only messages should be in the final result since LLM step was last
        assert "messages" in result
        assert len(result["messages"]) == 2  # Original + AI response


@pytest.mark.asyncio
async def test_agent_without_llm():
    """Test that Agent can be built without an LLM."""

    async def step(state):  # noqa: ARG001
        return {"processed": True}

    agent = await AgentFactory(SimpleState).add(step).build()

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

    agent = (
        await AgentFactory(ChatState).with_chat(mock_llm1).with_chat(mock_llm2).build()
    )

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

        agent = await AgentFactory(MixedState).add(step1).with_chat().build()

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

        agent = await (
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
        agent = await result3.build()

        assert agent is not None
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_unsupported_provider():
    """Test that unsupported LLM providers raise an error."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_chat_openai.side_effect = Exception("Unsupported provider")

        agent = await (
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

        agent = await AgentFactory(MixedState).add(step).with_chat().build()

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

        agent = await AgentFactory(MixedState).add(step).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_agent_run_without_llm():
    """Test that agent works without LLM steps."""

    async def step(state):  # noqa: ARG001
        return {"processed": True}

    agent = await AgentFactory(SimpleState).add(step).build()

    result = await agent.arun({})
    # Only the last step's fields should be in the result
    assert result["processed"] is True


@pytest.mark.asyncio
async def test_agent_with_prompt_template():
    """Test that prompt templates work correctly."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = await (
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

        agent = await (
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

        agent = await AgentFactory(ChatState).with_chat().build()

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

        agent = await AgentFactory(state_type=MyState).add(step).with_chat().build()

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

        agent = await AgentFactory(ChatState).with_chat().build()

        result = await agent.arun({"messages": []})
        assert "messages" in result


@pytest.mark.asyncio
async def test_agent_with_no_messages_key():
    """Test that agent handles missing messages key correctly."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = await AgentFactory(ChatState).with_chat().build()

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

        agent = await AgentFactory(MixedState).add(step).with_chat().build()

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

        agent = (
            await AgentFactory(CustomStateWithMessages).add(step).with_chat().build()
        )

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

        agent = (
            await AgentFactory(CustomStateWithoutMessages).add(step).with_chat().build()
        )

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
            await AgentFactory(None).add(step).with_chat().build()  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_agent_with_none_state_type_without_chat():
    """Test that None state type without chat raises an error since state_type is required."""

    async def step(state):  # noqa: ARG001
        return {"custom_field": "updated"}

    # Since state_type is now required, passing None should raise an error
    with pytest.raises(TypeError):
        await AgentFactory(None).add(step).build()  # type: ignore[arg-type]
