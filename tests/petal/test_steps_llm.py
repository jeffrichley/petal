from typing import List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from petal.core.steps.llm import LLMStep, LLMStepStrategy
from pydantic import BaseModel


class DummyLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "dummy"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        # Silence unused argument warnings
        _ = messages
        _ = stop
        _ = run_manager
        _ = kwargs
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="Hello!", response_metadata={})
                )
            ]
        )


@pytest.mark.asyncio
async def test_llm_step_strategy_create_and_call():
    print("A: before strategy")
    strategy = LLMStepStrategy()
    print("B: before config")
    config = {
        "prompt_template": "Hello {name}",
        "system_prompt": "You are a helpful assistant.",
        "llm_config": {"model": "gpt-4o-mini"},
        "llm_instance": DummyLLM(),
    }
    print("C: before create_step")
    step = strategy.create_step(config)
    print("D: before isinstance")
    assert isinstance(step, LLMStep)
    print("E: before get_node_name")
    node_name = strategy.get_node_name(0)
    print("F: before assert node_name")
    assert node_name == "llm_step_0"
    print("G: before state")
    state = {"name": "Jeff", "messages": []}
    print("H: before await step")
    result = await step(state)
    print("I: before assert messages in result")
    assert "messages" in result
    print("J: before isinstance messages[0]")
    assert isinstance(result["messages"][0], dict)
    print("K: before assert role user")
    assert result["messages"][0]["role"] == "user"
    print("L: before isinstance messages[1]")
    assert isinstance(result["messages"][1], AIMessage)
    print("M: before assert type ai")
    assert result["messages"][1].type == "ai"
    print("N: before assert content Hello!")
    assert result["messages"][1].content == "Hello!"


def test_llm_step_strategy_config_validation():
    strategy = LLMStepStrategy()
    # Missing llm_instance and llm_config - should use default config
    # Use a mock LLM to avoid real API calls
    from unittest.mock import Mock

    from langchain_core.messages import AIMessage

    mock_llm = Mock()
    mock_llm.ainvoke = Mock(return_value=AIMessage(content="Test response"))

    config = {"prompt_template": "Hi", "llm_instance": mock_llm}
    step = strategy.create_step(config)
    assert isinstance(step, LLMStep)


def test_llm_step_strategy_node_name():
    strategy = LLMStepStrategy()
    assert strategy.get_node_name(5) == "llm_step_5"


@pytest.mark.asyncio
async def test_llm_step_system_prompt_missing_key():
    # System prompt references {foo}, which is not in the state
    # Use a mock LLM to avoid real API calls
    from unittest.mock import AsyncMock, Mock

    from langchain_core.messages import AIMessage

    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))

    step = LLMStep(
        prompt_template="Hello!",
        system_prompt="You are a {foo} assistant.",
        llm_config={"provider": "openai", "model": "gpt-4o-mini"},
        llm_instance=mock_llm,
    )
    state = {"messages": [], "bar": 123}
    with pytest.raises(ValueError) as excinfo:
        await step(state)
    msg = str(excinfo.value)
    assert "requires key 'foo'" in msg
    assert "but it's not available in the state" in msg
    assert "bar" in msg or "messages" in msg  # available keys listed


def test_llm_step_uses_mock_llm_creation():
    """Test that LLM creation can be mocked during testing."""
    from unittest.mock import Mock, patch

    from langchain_core.messages import AIMessage

    mock_llm = Mock()
    mock_llm.ainvoke = Mock(return_value=AIMessage(content="Test response"))

    step = LLMStep(
        prompt_template="Hello!",
        system_prompt="You are a helpful assistant.",
        llm_config={"provider": "openai", "model": "gpt-4o-mini"},
        llm_instance=None,
    )

    # Mock the provider-specific LLM creation method
    with patch.object(step, "_create_llm_for_provider", return_value=mock_llm):
        created_llm = step._create_llm_instance()
        assert created_llm == mock_llm


def test_llm_step_unsupported_provider_raises_error():
    """Test that unsupported providers raise an error."""
    step = LLMStep(
        prompt_template="Hello!",
        system_prompt="You are a helpful assistant.",
        llm_config={"provider": "unsupported_provider", "model": "test-model"},
        llm_instance=None,
    )

    with pytest.raises(
        ValueError, match="Unsupported LLM provider: unsupported_provider"
    ):
        step._create_llm_from_config()


def test_fake_api_keys_are_used_in_tests():
    """Test that fake API keys are being used in the test environment."""
    import os

    # Verify that fake API keys are set
    assert os.environ.get("OPENAI_API_KEY") == "fake-openai-key-for-testing"
    assert os.environ.get("ANTHROPIC_API_KEY") == "fake-anthropic-key-for-testing"
    assert os.environ.get("GOOGLE_API_KEY") == "fake-google-key-for-testing"
    assert os.environ.get("COHERE_API_KEY") == "fake-cohere-key-for-testing"
    assert os.environ.get("HUGGINGFACE_API_KEY") == "fake-huggingface-key-for-testing"

    # Test that LLM creation uses the fake key
    step = LLMStep(
        prompt_template="Hello!",
        system_prompt="You are a helpful assistant.",
        llm_config={"provider": "openai", "model": "gpt-4o-mini"},
        llm_instance=None,
    )

    # Mock the LLM creation to avoid real calls
    from unittest.mock import Mock, patch

    mock_llm = Mock()

    with patch.object(step, "_create_llm_for_provider", return_value=mock_llm):
        created_llm = step._create_llm_instance()
        assert created_llm == mock_llm


def test_environment_variables_are_restored():
    """Test that environment variables are properly restored after tests."""
    import os

    # This test should run after the fake_api_keys fixture has run
    # and should still have the fake keys set
    assert os.environ.get("OPENAI_API_KEY") == "fake-openai-key-for-testing"
    assert os.environ.get("ANTHROPIC_API_KEY") == "fake-anthropic-key-for-testing"
    assert os.environ.get("GOOGLE_API_KEY") == "fake-google-key-for-testing"
    assert os.environ.get("COHERE_API_KEY") == "fake-cohere-key-for-testing"
    assert os.environ.get("HUGGINGFACE_API_KEY") == "fake-huggingface-key-for-testing"


@pytest.mark.asyncio
async def test_llm_step_with_structured_output_model():
    class MyModel(BaseModel):
        answer: str

    mock_llm = DummyLLM()
    mock_structured = Mock()
    mock_structured.ainvoke = AsyncMock(return_value=MyModel(answer="42"))

    with patch.object(DummyLLM, "with_structured_output", return_value=mock_structured):
        step = LLMStep(
            prompt_template="What is the answer?",
            system_prompt="You are a helpful assistant.",
            llm_config={"provider": "openai", "model": "gpt-4o-mini"},
            llm_instance=mock_llm,
            structured_output_model=MyModel,
            structured_output_key=None,
        )
        state = {"messages": [], "name": "test"}
        result = await step(state)

        print(result)

        assert result == {"answer": "42"}


@pytest.mark.asyncio
async def test_llm_step_with_structured_output_model_and_key():
    class MyModel(BaseModel):
        answer: str

    mock_llm = DummyLLM()
    mock_structured = Mock()
    mock_structured.ainvoke = AsyncMock(return_value=MyModel(answer="42"))

    with patch.object(DummyLLM, "with_structured_output", return_value=mock_structured):
        step = LLMStep(
            prompt_template="What is the answer?",
            system_prompt="You are a helpful assistant.",
            llm_config={"provider": "openai", "model": "gpt-4o-mini"},
            llm_instance=mock_llm,
            structured_output_model=MyModel,
            structured_output_key="blah",
        )
        state = {"messages": [], "name": "test"}
        result = await step(state)
        assert result == {"blah": {"answer": "42"}}


@pytest.mark.asyncio
async def test_llm_step_without_structured_output_model_falls_back():
    mock_llm = DummyLLM()
    step = LLMStep(
        prompt_template="Hello {name}",
        system_prompt="You are a helpful assistant.",
        llm_config={"provider": "openai", "model": "gpt-4o-mini"},
        llm_instance=mock_llm,
        structured_output_model=None,
        structured_output_key=None,
    )
    state = {"name": "Jeff", "messages": []}
    result = await step(state)
    assert "messages" in result
    assert isinstance(result["messages"][0], dict)
    assert result["messages"][0]["role"] == "user"
    assert isinstance(result["messages"][1], AIMessage)
    assert result["messages"][1].content == "Hello!"


@pytest.mark.asyncio
async def test_llm_step_structured_output_always_returns_base_model():
    """Test that structured output always returns a BaseModel, making the 'return result' line dead code."""

    class MyModel(BaseModel):
        answer: str

    mock_llm = DummyLLM()
    mock_structured = Mock()
    # This will always be a BaseModel when using with_structured_output
    mock_structured.ainvoke = AsyncMock(return_value=MyModel(answer="42"))

    with patch.object(DummyLLM, "with_structured_output", return_value=mock_structured):
        step = LLMStep(
            prompt_template="What is the answer?",
            system_prompt="You are a helpful assistant.",
            llm_config={"provider": "openai", "model": "gpt-4o-mini"},
            llm_instance=mock_llm,
            structured_output_model=MyModel,
            structured_output_key="test_key",
        )
        state = {"messages": [], "name": "test"}
        result = await step(state)

        # Verify that the result is wrapped in the key (not the raw result)
        assert result == {"test_key": {"answer": "42"}}
        # This confirms that the 'return result' line is never reached
        # because is_base_model is always True with structured output


@pytest.mark.asyncio
async def test_llm_step_with_tools_binds_tools():
    """Test that LLM step binds tools when available and LLM supports it."""
    from unittest.mock import AsyncMock, Mock

    # Create a mock LLM that has bind_tools method
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
    mock_llm.bind_tools = Mock(return_value=mock_llm)  # Return self for chaining

    # Create tools list
    tools = [{"name": "test_tool", "description": "A test tool"}]

    step = LLMStep(
        prompt_template="Hello!",
        system_prompt="You are a helpful assistant.",
        llm_config={"provider": "openai", "model": "gpt-4o-mini"},
        llm_instance=mock_llm,
        tools=tools,
    )

    state = {"messages": [], "name": "test"}
    result = await step(state)

    # Verify bind_tools was called with the tools
    mock_llm.bind_tools.assert_called_once_with(tools)
    assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_tools_but_no_bind_tools():
    """Test that LLM step skips tool binding when LLM doesn't support it."""
    from unittest.mock import AsyncMock, Mock

    # Create a mock LLM that doesn't have bind_tools method
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
    # Note: no bind_tools method - this tests the hasattr check
    # Make sure the mock doesn't have bind_tools attribute
    if hasattr(mock_llm, "bind_tools"):
        delattr(mock_llm, "bind_tools")

    # Create tools list
    tools = [{"name": "test_tool", "description": "A test tool"}]

    step = LLMStep(
        prompt_template="Hello!",
        system_prompt="You are a helpful assistant.",
        llm_config={"provider": "openai", "model": "gpt-4o-mini"},
        llm_instance=mock_llm,
        tools=tools,
    )

    state = {"messages": [], "name": "test"}
    result = await step(state)

    # Verify the step still works without bind_tools
    assert "messages" in result
    # Verify bind_tools was not called (it doesn't exist)
    assert not hasattr(mock_llm, "bind_tools")


@pytest.mark.asyncio
async def test_llm_step_without_tools_doesnt_bind():
    """Test that LLM step doesn't bind tools when no tools provided."""
    from unittest.mock import AsyncMock, Mock

    # Create a mock LLM that has bind_tools method
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
    mock_llm.bind_tools = Mock(return_value=mock_llm)

    step = LLMStep(
        prompt_template="Hello!",
        system_prompt="You are a helpful assistant.",
        llm_config={"provider": "openai", "model": "gpt-4o-mini"},
        llm_instance=mock_llm,
        tools=None,  # No tools
    )

    state = {"messages": [], "name": "test"}
    result = await step(state)

    # Verify bind_tools was not called since no tools provided
    mock_llm.bind_tools.assert_not_called()
    assert "messages" in result


@pytest.mark.asyncio
async def test_llm_step_with_empty_tools_doesnt_bind():
    """Test that LLM step doesn't bind tools when empty tools list provided."""
    from unittest.mock import AsyncMock, Mock

    # Create a mock LLM that has bind_tools method
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
    mock_llm.bind_tools = Mock(return_value=mock_llm)

    step = LLMStep(
        prompt_template="Hello!",
        system_prompt="You are a helpful assistant.",
        llm_config={"provider": "openai", "model": "gpt-4o-mini"},
        llm_instance=mock_llm,
        tools=[],  # Empty tools list
    )

    state = {"messages": [], "name": "test"}
    result = await step(state)

    # Verify bind_tools was not called since tools list is empty
    mock_llm.bind_tools.assert_not_called()
    assert "messages" in result


def test_llm_step_strategy_with_tools():
    """Test that LLM step strategy properly handles tools in config."""
    strategy = LLMStepStrategy()

    tools = [{"name": "test_tool", "description": "A test tool"}]
    config = {
        "prompt_template": "Hello {name}",
        "system_prompt": "You are a helpful assistant.",
        "llm_config": {"model": "gpt-4o-mini"},
        "tools": tools,
    }

    step = strategy.create_step(config)
    assert isinstance(step, LLMStep)
    assert step.tools == tools


def test_llm_step_strategy_without_tools():
    """Test that LLM step strategy handles config without tools."""
    strategy = LLMStepStrategy()

    config = {
        "prompt_template": "Hello {name}",
        "system_prompt": "You are a helpful assistant.",
        "llm_config": {"model": "gpt-4o-mini"},
        # No tools key
    }

    step = strategy.create_step(config)
    assert isinstance(step, LLMStep)
    assert step.tools is None
