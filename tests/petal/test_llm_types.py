"""Tests for LLMTypes preconfigured configurations."""

import pytest

from petal.core.config.agent import LLMConfig
from petal.core.config.llm_types import LLMTypes
from petal.core.factory import AgentFactory


class TestLLMTypes:
    """Test LLMTypes preconfigured configurations."""

    def test_openai_gpt4o_mini_config(self):
        """Test OpenAI GPT-4o-mini configuration."""
        config = LLMTypes.OPENAI_GPT4O_MINI
        assert isinstance(config, LLMConfig)
        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 8000

    def test_openai_gpt4o_config(self):
        config = LLMTypes.OPENAI_GPT4O
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.0
        assert config.max_tokens == 8000

    def test_openai_gpt4o_large_config(self):
        config = LLMTypes.OPENAI_GPT4O_LARGE
        assert config.provider == "openai"
        assert config.model == "gpt-4o-large"
        assert config.temperature == 0.0
        assert config.max_tokens == 8000

    def test_openai_gpt35_turbo_config(self):
        config = LLMTypes.OPENAI_GPT35_TURBO
        assert config.provider == "openai"
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.0
        assert config.max_tokens == 8000

    def test_creative_configurations(self):
        creative_configs = [
            LLMTypes.OPENAI_GPT4O_CREATIVE,
            LLMTypes.OPENAI_GPT4O_MINI_CREATIVE,
        ]
        for config in creative_configs:
            assert config.temperature == 0.7
            assert config.max_tokens == 8000

    def test_analytical_configurations(self):
        analytical_configs = [
            LLMTypes.OPENAI_GPT4O_ANALYTICAL,
            LLMTypes.OPENAI_GPT4O_MINI_ANALYTICAL,
        ]
        for config in analytical_configs:
            assert config.temperature == 0.1
            assert config.max_tokens == 8000

    def test_anthropic_configurations(self):
        haiku_config = LLMTypes.ANTHROPIC_CLAUDE_3_HAIKU
        sonnet_config = LLMTypes.ANTHROPIC_CLAUDE_3_SONNET
        opus_config = LLMTypes.ANTHROPIC_CLAUDE_3_OPUS
        assert haiku_config.provider == "anthropic"
        assert haiku_config.model == "claude-3-haiku-20240307"
        assert sonnet_config.provider == "anthropic"
        assert sonnet_config.model == "claude-3-sonnet-20240229"
        assert opus_config.provider == "anthropic"
        assert opus_config.model == "claude-3-opus-20240229"

    def test_google_configurations(self):
        gemini_pro = LLMTypes.GOOGLE_GEMINI_PRO
        gemini_creative = LLMTypes.GOOGLE_GEMINI_PRO_CREATIVE
        assert gemini_pro.provider == "google"
        assert gemini_pro.model == "gemini-pro"
        assert gemini_pro.temperature == 0.0
        assert gemini_creative.provider == "google"
        assert gemini_creative.model == "gemini-pro"
        assert gemini_creative.temperature == 0.7

    def test_cohere_configurations(self):
        command = LLMTypes.COHERE_COMMAND
        command_creative = LLMTypes.COHERE_COMMAND_CREATIVE
        assert command.provider == "cohere"
        assert command.model == "command"
        assert command.temperature == 0.0
        assert command_creative.provider == "cohere"
        assert command_creative.model == "command"
        assert command_creative.temperature == 0.7

    def test_create_custom_basic(self):
        config = LLMTypes.create_custom(
            provider="openai", model="gpt-4o", temperature=0.5, max_tokens=8000
        )
        assert isinstance(config, LLMConfig)
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_tokens == 8000

    def test_create_custom_with_extra_params(self):
        config = LLMTypes.create_custom(
            provider="openai", model="gpt-4o", temperature=0.3, max_tokens=8000
        )
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.3
        assert config.max_tokens == 8000

    def test_with_temperature(self):
        base_config = LLMTypes.OPENAI_GPT4O
        creative_config = LLMTypes.with_temperature(base_config, 0.8)
        assert creative_config is not base_config
        assert creative_config.temperature == 0.8
        assert creative_config.provider == base_config.provider
        assert creative_config.model == base_config.model
        assert creative_config.max_tokens == base_config.max_tokens

    def test_with_max_tokens(self):
        base_config = LLMTypes.OPENAI_GPT4O
        long_response_config = LLMTypes.with_max_tokens(base_config, 8000)
        assert long_response_config is not base_config
        assert long_response_config.max_tokens == 8000
        assert long_response_config.provider == base_config.provider
        assert long_response_config.model == base_config.model
        assert long_response_config.temperature == base_config.temperature

    def test_with_temperature_and_max_tokens_chain(self):
        base_config = LLMTypes.OPENAI_GPT4O
        custom_config = LLMTypes.with_temperature(
            LLMTypes.with_max_tokens(base_config, 8000), 0.6
        )
        assert custom_config.temperature == 0.6
        assert custom_config.max_tokens == 8000
        assert custom_config.provider == "openai"
        assert custom_config.model == "gpt-4o"


class TestLLMTypesIntegration:
    """Test LLMTypes integration with AgentFactory."""

    @pytest.mark.asyncio
    async def test_llm_types_with_agent_factory(self):
        from langgraph.graph.message import add_messages
        from typing_extensions import Annotated, TypedDict

        class TestState(TypedDict):
            messages: Annotated[list, add_messages]
            test_field: str

        agent = (
            AgentFactory(TestState)
            .with_chat(
                llm_config=LLMTypes.OPENAI_GPT4O_MINI, prompt_template="Test prompt"
            )
            .build()
        )
        assert agent is not None

    @pytest.mark.asyncio
    async def test_llm_types_creative_with_agent_factory(self):
        from langgraph.graph.message import add_messages
        from typing_extensions import Annotated, TypedDict

        class TestState(TypedDict):
            messages: Annotated[list, add_messages]
            test_field: str

        agent = (
            AgentFactory(TestState)
            .with_chat(
                llm_config=LLMTypes.OPENAI_GPT4O_CREATIVE,
                prompt_template="Be creative!",
            )
            .build()
        )
        assert agent is not None

    @pytest.mark.asyncio
    async def test_llm_types_analytical_with_agent_factory(self):
        from langgraph.graph.message import add_messages
        from typing_extensions import Annotated, TypedDict

        class TestState(TypedDict):
            messages: Annotated[list, add_messages]
            test_field: str

        agent = (
            AgentFactory(TestState)
            .with_chat(
                llm_config=LLMTypes.OPENAI_GPT4O_ANALYTICAL,
                prompt_template="Analyze this carefully.",
            )
            .build()
        )
        assert agent is not None

    @pytest.mark.asyncio
    async def test_custom_llm_config_with_agent_factory(self):
        from langgraph.graph.message import add_messages
        from typing_extensions import Annotated, TypedDict

        class TestState(TypedDict):
            messages: Annotated[list, add_messages]
            test_field: str

        custom_config = LLMTypes.create_custom(
            "openai", "gpt-4o", temperature=0.3, max_tokens=8000
        )
        agent = (
            AgentFactory(TestState)
            .with_chat(
                llm_config=custom_config, prompt_template="Custom configuration test"
            )
            .build()
        )
        assert agent is not None
