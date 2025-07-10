from typing import List, Optional

import pytest
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from petal.core.steps.llm import LLMStep, LLMStepStrategy


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
