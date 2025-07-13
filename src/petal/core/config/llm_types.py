"""Preconfigured LLM configurations for easy use with AgentFactory."""

from petal.core.config.agent import LLMConfig


class PreconfiguredLLMConfig(LLMConfig):
    """A Pydantic model for preconfigured LLM settings."""

    pass


class LLMTypes:
    """
    Preconfigured LLM configurations for common use cases.

    Usage:
        AgentFactory(MyState).with_chat(
            llm_config=LLMTypes.OPENAI_GPT4O_MINI.model_dump(),
            prompt_template="Your prompt here"
        )
    """

    # OpenAI configurations
    OPENAI_GPT4O_MINI = PreconfiguredLLMConfig(
        provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=8000
    )
    OPENAI_GPT4O = PreconfiguredLLMConfig(
        provider="openai", model="gpt-4o", temperature=0.0, max_tokens=8000
    )
    OPENAI_GPT4O_LARGE = PreconfiguredLLMConfig(
        provider="openai", model="gpt-4o-large", temperature=0.0, max_tokens=8000
    )
    OPENAI_GPT35_TURBO = PreconfiguredLLMConfig(
        provider="openai", model="gpt-3.5-turbo", temperature=0.0, max_tokens=8000
    )
    # Creative configurations (higher temperature)
    OPENAI_GPT4O_CREATIVE = PreconfiguredLLMConfig(
        provider="openai", model="gpt-4o", temperature=0.7, max_tokens=8000
    )
    OPENAI_GPT4O_MINI_CREATIVE = PreconfiguredLLMConfig(
        provider="openai", model="gpt-4o-mini", temperature=0.7, max_tokens=8000
    )
    # Analytical configurations (lower temperature)
    OPENAI_GPT4O_ANALYTICAL = PreconfiguredLLMConfig(
        provider="openai", model="gpt-4o", temperature=0.1, max_tokens=8000
    )
    OPENAI_GPT4O_MINI_ANALYTICAL = PreconfiguredLLMConfig(
        provider="openai", model="gpt-4o-mini", temperature=0.1, max_tokens=8000
    )
    # Anthropic configurations (when supported)
    ANTHROPIC_CLAUDE_3_HAIKU = PreconfiguredLLMConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        temperature=0.0,
        max_tokens=8000,
    )
    ANTHROPIC_CLAUDE_3_SONNET = PreconfiguredLLMConfig(
        provider="anthropic",
        model="claude-3-sonnet-20240229",
        temperature=0.0,
        max_tokens=8000,
    )
    ANTHROPIC_CLAUDE_3_OPUS = PreconfiguredLLMConfig(
        provider="anthropic",
        model="claude-3-opus-20240229",
        temperature=0.0,
        max_tokens=8000,
    )
    # Google configurations (when supported)
    GOOGLE_GEMINI_PRO = PreconfiguredLLMConfig(
        provider="google", model="gemini-pro", temperature=0.0, max_tokens=8000
    )
    GOOGLE_GEMINI_PRO_CREATIVE = PreconfiguredLLMConfig(
        provider="google", model="gemini-pro", temperature=0.7, max_tokens=8000
    )
    # Cohere configurations (when supported)
    COHERE_COMMAND = PreconfiguredLLMConfig(
        provider="cohere", model="command", temperature=0.0, max_tokens=8000
    )
    COHERE_COMMAND_CREATIVE = PreconfiguredLLMConfig(
        provider="cohere", model="command", temperature=0.7, max_tokens=8000
    )
    # Ollama configurations (local LLM support)
    OLLAMA_LLAMA2 = PreconfiguredLLMConfig(
        provider="ollama", model="llama2", temperature=0.0, max_tokens=8000
    )
    OLLAMA_LLAMA2_CREATIVE = PreconfiguredLLMConfig(
        provider="ollama", model="llama2", temperature=0.7, max_tokens=8000
    )
    OLLAMA_MISTRAL = PreconfiguredLLMConfig(
        provider="ollama", model="mistral", temperature=0.0, max_tokens=8000
    )
    OLLAMA_MISTRAL_CREATIVE = PreconfiguredLLMConfig(
        provider="ollama", model="mistral", temperature=0.7, max_tokens=8000
    )
    OLLAMA_CODESTRAL = PreconfiguredLLMConfig(
        provider="ollama", model="codestral", temperature=0.0, max_tokens=8000
    )
    OLLAMA_CODESTRAL_CREATIVE = PreconfiguredLLMConfig(
        provider="ollama", model="codestral", temperature=0.7, max_tokens=8000
    )
    OLLAMA_NEURAL_CHAT = PreconfiguredLLMConfig(
        provider="ollama", model="neural-chat", temperature=0.0, max_tokens=8000
    )
    OLLAMA_NEURAL_CHAT_CREATIVE = PreconfiguredLLMConfig(
        provider="ollama", model="neural-chat", temperature=0.7, max_tokens=8000
    )

    @classmethod
    def create_custom(
        cls,
        provider: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8000,
        **kwargs,
    ) -> PreconfiguredLLMConfig:
        """
        Create a custom LLM configuration.
        Returns a PreconfiguredLLMConfig (Pydantic model).
        """
        return PreconfiguredLLMConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @classmethod
    def with_temperature(
        cls, base_config: PreconfiguredLLMConfig, temperature: float
    ) -> PreconfiguredLLMConfig:
        """
        Return a copy of the config with a new temperature.
        """
        return base_config.model_copy(update={"temperature": temperature})

    @classmethod
    def with_max_tokens(
        cls, base_config: PreconfiguredLLMConfig, max_tokens: int
    ) -> PreconfiguredLLMConfig:
        """
        Return a copy of the config with a new max_tokens value.
        """
        return base_config.model_copy(update={"max_tokens": max_tokens})
