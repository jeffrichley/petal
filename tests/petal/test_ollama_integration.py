"""Tests for Ollama integration with Petal framework."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from petal.core.config.llm_types import LLMTypes
from petal.core.factory import AgentFactory
from petal.core.steps.llm import LLMStep


class TestOllamaIntegration:
    """Test Ollama integration functionality."""

    def test_ollama_provider_validation(self):
        """Test that 'ollama' is a valid provider."""
        from petal.core.config.agent import validate_provider

        # Should not raise an exception
        result = validate_provider("ollama")
        assert result == "ollama"

    def test_ollama_preconfigured_configs(self):
        """Test that Ollama preconfigured configs are available."""
        # Test basic configs
        assert hasattr(LLMTypes, "OLLAMA_LLAMA2")
        assert hasattr(LLMTypes, "OLLAMA_MISTRAL")
        assert hasattr(LLMTypes, "OLLAMA_CODESTRAL")
        assert hasattr(LLMTypes, "OLLAMA_NEURAL_CHAT")

        # Test creative configs
        assert hasattr(LLMTypes, "OLLAMA_LLAMA2_CREATIVE")
        assert hasattr(LLMTypes, "OLLAMA_MISTRAL_CREATIVE")
        assert hasattr(LLMTypes, "OLLAMA_CODESTRAL_CREATIVE")
        assert hasattr(LLMTypes, "OLLAMA_NEURAL_CHAT_CREATIVE")

        # Verify config properties
        llama2_config = LLMTypes.OLLAMA_LLAMA2
        assert llama2_config.provider == "ollama"
        assert llama2_config.model == "llama2"
        assert llama2_config.temperature == 0.0
        assert llama2_config.max_tokens == 8000

        # Verify creative config has higher temperature
        llama2_creative_config = LLMTypes.OLLAMA_LLAMA2_CREATIVE
        assert llama2_creative_config.provider == "ollama"
        assert llama2_creative_config.model == "llama2"
        assert llama2_creative_config.temperature == 0.7

    def test_ollama_llm_step_creation(self):
        """Test that LLMStep can create Ollama instances."""

        # Mock the ChatOllama class
        with patch("petal.core.steps.llm.ChatOllama") as mock_chat_ollama:
            mock_instance = MagicMock()
            mock_chat_ollama.return_value = mock_instance

            # Create LLM step with Ollama config
            step = LLMStep(
                prompt_template="Test prompt",
                system_prompt="Test system prompt",
                llm_config={
                    "provider": "ollama",
                    "model": "llama2",
                    "temperature": 0.5,
                    "max_tokens": 1000,
                },
                llm_instance=None,
            )

            # Call the method that creates the LLM instance
            step._create_llm_from_config()

            # Verify ChatOllama was called with correct parameters
            mock_chat_ollama.assert_called_once()
            call_args = mock_chat_ollama.call_args[1]  # kwargs
            assert call_args["model"] == "llama2"
            assert call_args["temperature"] == 0.5
            assert call_args["max_tokens"] == 1000
            assert call_args["base_url"] == "http://localhost:11434"

    def test_ollama_with_custom_base_url(self):
        """Test that custom base_url is respected for Ollama."""
        with patch("petal.core.steps.llm.ChatOllama") as mock_chat_ollama:
            mock_instance = MagicMock()
            mock_chat_ollama.return_value = mock_instance

            step = LLMStep(
                prompt_template="Test prompt",
                system_prompt="Test system prompt",
                llm_config={
                    "provider": "ollama",
                    "model": "mistral",
                    "base_url": "http://custom-ollama:11434",
                },
                llm_instance=None,
            )

            step._create_llm_from_config()

            # Verify custom base_url was used
            call_args = mock_chat_ollama.call_args[1]
            assert call_args["base_url"] == "http://custom-ollama:11434"

    @pytest.mark.asyncio
    async def test_ollama_agent_factory_integration(self):
        """Test that AgentFactory can create agents with Ollama."""
        from typing import Annotated, TypedDict

        from langgraph.graph.message import add_messages

        class TestState(TypedDict):
            messages: Annotated[list, add_messages]
            user_input: str

        # Mock the LLM step to avoid actual Ollama calls
        with patch(
            "petal.core.steps.llm.LLMStep._create_llm_from_config"
        ) as mock_create_llm:
            mock_llm = AsyncMock()
            mock_create_llm.return_value = mock_llm

            # Mock the LLM response with proper message format
            from langchain_core.messages import AIMessage

            mock_response = AIMessage(content="Hello from Ollama!")
            mock_llm.ainvoke.return_value = mock_response

            # Create agent with Ollama
            agent = await (
                AgentFactory(TestState)
                .with_chat(
                    prompt_template="User says: {user_input}",
                    system_prompt="You are a helpful assistant.",
                    llm_config={
                        "provider": "ollama",
                        "model": "llama2",
                        "temperature": 0.0,
                    },
                )
                .build()
            )

            # Test the agent
            result = await agent.arun({"user_input": "Hello!", "messages": []})

            # Verify the response
            assert "messages" in result
            assert len(result["messages"]) > 0
            assert result["messages"][-1].content == "Hello from Ollama!"

    def test_ollama_custom_config_creation(self):
        """Test creating custom Ollama configurations."""
        custom_config = LLMTypes.create_custom(
            provider="ollama",
            model="custom-model",
            temperature=0.3,
            max_tokens=1500,
        )

        assert custom_config.provider == "ollama"
        assert custom_config.model == "custom-model"
        assert custom_config.temperature == 0.3
        assert custom_config.max_tokens == 1500

    def test_ollama_temperature_modification(self):
        """Test modifying temperature for Ollama configs."""
        base_config = LLMTypes.OLLAMA_LLAMA2
        modified_config = LLMTypes.with_temperature(base_config, 0.8)

        assert modified_config.provider == "ollama"
        assert modified_config.model == "llama2"
        assert modified_config.temperature == 0.8
        assert modified_config.max_tokens == 8000

    def test_ollama_max_tokens_modification(self):
        """Test modifying max_tokens for Ollama configs."""
        base_config = LLMTypes.OLLAMA_MISTRAL
        modified_config = LLMTypes.with_max_tokens(base_config, 5000)

        assert modified_config.provider == "ollama"
        assert modified_config.model == "mistral"
        assert modified_config.temperature == 0.0
        assert modified_config.max_tokens == 5000
