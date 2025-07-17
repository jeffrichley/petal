from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from petal.core.factory import AgentFactory

from tests.petal.conftest_factory import (
    ChatState,
    DefaultState,
    MixedState,
    SimpleState,
)


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

        agent = await (
            AgentFactory(PromptState)
            .with_chat()
            .with_prompt("Process {input_data} (step {step_count})")
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

        agent = await (
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

        agent = await AgentFactory(ChatState).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_empty_messages():
    """Test that LLM step handles empty messages correctly."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = await AgentFactory(ChatState).with_chat().build()

        result = await agent.arun({"messages": []})
        assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_missing_messages_key():
    """Test that LLM step handles missing messages key correctly."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_chat_openai.return_value = mock_llm

        agent = await AgentFactory(ChatState).with_chat().build()

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

        agent = await AgentFactory(ChatState).with_chat().build()
        state = {"messages": [HumanMessage(content="test")], "__async__": True}
        result = await agent.arun(state)
        assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_custom_llm_instance():
    """Test that LLM step works with custom LLM instance."""
    from langchain.chat_models.base import BaseChatModel

    mock_llm = Mock(spec=BaseChatModel)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))

    agent = await AgentFactory(ChatState).with_chat(mock_llm).build()

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
        agent = await AgentFactory(ChatState).with_chat(llm_config=config).build()

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

        agent = await AgentFactory(ChatState).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        mock_chat_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0)
        assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_unsupported_provider():
    """Test that LLM step raises error for unsupported provider."""
    config = {"provider": "unsupported", "model": "test"}
    agent = await AgentFactory(ChatState).with_chat(llm_config=config).build()
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

        agent = await AgentFactory(ChatState).with_chat().with_chat().build()

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

        agent = await AgentFactory(MixedState).add(regular_step).with_chat().build()

        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_agent_build_with_custom_node_names():
    """Test that building agent with custom node names works correctly."""

    async def step1(state):  # noqa: ARG001
        return {"x": 1}

    async def step2(state):  # noqa: ARG001
        return {"x": state.get("x", 0) + 2}

    agent = await (
        AgentFactory(SimpleState)
        .add(step1, "custom_step_1")
        .add(step2, "custom_step_2")
        .build()
    )

    result = await agent.arun({})
    assert result["x"] == 3


@pytest.mark.asyncio
async def test_llm_step_prompt_template_missing_key():
    """Test that a helpful ValueError is raised if the prompt template references a missing key."""
    with patch("petal.core.steps.llm.ChatOpenAI") as mock_chat_openai:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Test response")
        mock_chat_openai.return_value = mock_llm

        agent = await (
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

        agent = (
            await factory.with_chat(llm=mock_llm).with_prompt("Hello {name}").build()
        )

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

        agent = (
            await factory.with_chat(llm=mock_llm).with_prompt("Hello {name}").build()
        )

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

        agent = await (
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

        agent = await (
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

        agent = await factory.with_chat(llm=mock_llm).with_prompt("Continue").build()
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
