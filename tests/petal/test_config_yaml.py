"""Tests for YAML configuration models."""

import pytest
from petal.core.config.yaml import (
    BaseNodeConfig,
    LLMNodeConfig,
    ReactNodeConfig,
    StateSchemaConfig,
    ValidationConfig,
    validate_max_iterations,
    validate_max_tokens,
    validate_node_type,
    validate_provider,
    validate_temperature,
)
from pydantic import ValidationError


class TestValidationFunctions:
    """Test validation functions."""

    def test_validate_node_type_valid(self):
        """Test valid node types."""
        assert validate_node_type("llm") == "llm"
        assert validate_node_type("react") == "react"

    def test_validate_node_type_invalid(self):
        """Test invalid node types."""
        with pytest.raises(ValueError, match="node_type must be one of"):
            validate_node_type("invalid")
        with pytest.raises(ValueError, match="node_type must be one of"):
            validate_node_type("custom")

    def test_validate_provider_valid(self):
        """Test valid providers."""
        assert validate_provider("openai") == "openai"
        assert validate_provider("anthropic") == "anthropic"
        assert validate_provider("google") == "google"
        assert validate_provider("cohere") == "cohere"
        assert validate_provider("huggingface") == "huggingface"

    def test_validate_provider_invalid(self):
        """Test invalid providers."""
        with pytest.raises(ValueError, match="provider must be one of"):
            validate_provider("invalid")
        with pytest.raises(ValueError, match="provider must be one of"):
            validate_provider("azure")

    def test_validate_temperature_valid(self):
        """Test valid temperature values."""
        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(1.0) == 1.0
        assert validate_temperature(2.0) == 2.0

    def test_validate_temperature_invalid(self):
        """Test invalid temperature values."""
        with pytest.raises(ValueError, match="temperature must be between"):
            validate_temperature(-0.1)
        with pytest.raises(ValueError, match="temperature must be between"):
            validate_temperature(2.1)

    def test_validate_max_tokens_valid(self):
        """Test valid max_tokens values."""
        assert validate_max_tokens(1) == 1
        assert validate_max_tokens(1000) == 1000
        assert validate_max_tokens(8000) == 8000

    def test_validate_max_tokens_invalid(self):
        """Test invalid max_tokens values."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            validate_max_tokens(0)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            validate_max_tokens(-1)

    def test_validate_max_iterations_valid(self):
        """Test valid max_iterations values."""
        assert validate_max_iterations(1) == 1
        assert validate_max_iterations(5) == 5
        assert validate_max_iterations(10) == 10

    def test_validate_max_iterations_invalid(self):
        """Test invalid max_iterations values."""
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            validate_max_iterations(0)
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            validate_max_iterations(-1)


class TestStateSchemaConfig:
    """Test StateSchemaConfig model."""

    def test_valid_config(self):
        """Test valid state schema configuration."""
        config = StateSchemaConfig(
            fields={"input": str, "output": str}, required_fields=["input"]
        )
        assert config.fields == {"input": str, "output": str}
        assert config.required_fields == ["input"]

    def test_default_config(self):
        """Test default state schema configuration."""
        config = StateSchemaConfig()
        assert config.fields == {}
        assert config.required_fields == []


class TestValidationConfig:
    """Test ValidationConfig model."""

    def test_valid_config(self):
        """Test valid validation configuration."""
        input_schema = StateSchemaConfig(fields={"input": str})
        output_schema = StateSchemaConfig(fields={"output": str})

        config = ValidationConfig(
            input_schema=input_schema, output_schema=output_schema
        )
        assert config.input_schema == input_schema
        assert config.output_schema == output_schema

    def test_default_config(self):
        """Test default validation configuration."""
        config = ValidationConfig(input_schema=None, output_schema=None)
        assert config.input_schema is None
        assert config.output_schema is None


class TestBaseNodeConfig:
    """Test BaseNodeConfig model."""

    def test_valid_config(self):
        """Test valid base node configuration."""
        config = BaseNodeConfig(type="llm", name="test_node", description="A test node")
        assert config.type == "llm"
        assert config.name == "test_node"
        assert config.description == "A test node"
        assert config.enabled is True

    def test_minimal_config(self):
        """Test minimal base node configuration."""
        config = BaseNodeConfig(type="react", name="minimal", description="")
        assert config.type == "react"
        assert config.name == "minimal"
        assert config.description == ""
        assert config.enabled is True

    def test_invalid_type(self):
        """Test invalid node type."""
        with pytest.raises(ValidationError):
            BaseNodeConfig(type="invalid", name="test", description="")

    def test_empty_name(self):
        """Test empty name validation."""
        with pytest.raises(ValidationError, match="name cannot be empty"):
            BaseNodeConfig(type="llm", name="", description="")
        with pytest.raises(ValidationError, match="name cannot be empty"):
            BaseNodeConfig(type="llm", name="   ", description="")

    def test_name_stripping(self):
        """Test name whitespace stripping."""
        config = BaseNodeConfig(type="llm", name="  test  ", description="")
        assert config.name == "test"


class TestLLMNodeConfig:
    """Test LLMNodeConfig model."""

    def test_valid_config(self):
        """Test valid LLM node configuration."""
        config = LLMNodeConfig(
            type="llm",
            name="assistant",
            description="A helpful assistant",
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=1000,
            prompt="Answer the user's question: {user_input}",
            system_prompt="You are a helpful assistant.",
        )
        assert config.type == "llm"
        assert config.name == "assistant"
        assert config.description == "A helpful assistant"
        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 1000
        assert config.prompt == "Answer the user's question: {user_input}"
        assert config.system_prompt == "You are a helpful assistant."

    def test_minimal_config(self):
        """Test minimal LLM node configuration."""
        config = LLMNodeConfig(
            type="llm",
            name="minimal",
            provider="anthropic",
            model="claude-3-haiku",
            description="",
            prompt=None,
            system_prompt=None,
        )
        assert config.type == "llm"
        assert config.name == "minimal"
        assert config.provider == "anthropic"
        assert config.model == "claude-3-haiku"
        assert config.temperature == 0.0
        assert config.max_tokens == 1000
        assert config.prompt is None
        assert config.system_prompt is None

    def test_invalid_provider(self):
        """Test invalid provider."""
        with pytest.raises(ValidationError, match="provider must be one of"):
            LLMNodeConfig(
                type="llm",
                name="test",
                provider="invalid",
                model="gpt-4",
                description="",
                prompt="",
                system_prompt="",
            )

    def test_invalid_temperature(self):
        """Test invalid temperature."""
        with pytest.raises(ValidationError, match="temperature must be between"):
            LLMNodeConfig(
                type="llm",
                name="test",
                provider="openai",
                model="gpt-4",
                temperature=2.1,
                description="",
                prompt="",
                system_prompt="",
            )

    def test_invalid_max_tokens(self):
        """Test invalid max_tokens."""
        with pytest.raises(ValidationError, match="max_tokens must be positive"):
            LLMNodeConfig(
                type="llm",
                name="test",
                provider="openai",
                model="gpt-4",
                max_tokens=0,
                description="",
                prompt="",
                system_prompt="",
            )

    def test_empty_model(self):
        """Test empty model validation."""
        with pytest.raises(ValidationError, match="model cannot be empty"):
            LLMNodeConfig(
                type="llm",
                name="test",
                provider="openai",
                model="",
                description="",
                prompt="",
                system_prompt="",
            )

    def test_empty_prompt(self):
        """Test empty prompt validation."""
        with pytest.raises(ValidationError, match="prompt cannot be empty if provided"):
            LLMNodeConfig(
                type="llm",
                name="test",
                provider="openai",
                model="gpt-4",
                prompt="",
                description="",
                system_prompt="",
            )

    def test_empty_system_prompt(self):
        """Test empty system_prompt validation."""
        with pytest.raises(
            ValidationError, match="system_prompt cannot be empty if provided"
        ):
            LLMNodeConfig(
                type="llm",
                name="test",
                provider="openai",
                model="gpt-4",
                system_prompt="",
                description="",
                prompt="",
            )

    def test_prompt_stripping(self):
        """Test prompt whitespace stripping."""
        config = LLMNodeConfig(
            type="llm",
            name="test",
            provider="openai",
            model="gpt-4",
            prompt="  test prompt  ",
            system_prompt="  test system  ",
            description="",
        )
        assert config.prompt == "test prompt"
        assert config.system_prompt == "test system"


class TestReactNodeConfig:
    """Test ReactNodeConfig model."""

    def test_valid_config(self):
        """Test valid React node configuration."""
        config = ReactNodeConfig(
            type="react",
            name="reasoning_agent",
            description="An agent that can use tools and reason",
            tools=["search", "calculator", "database"],
            reasoning_prompt="Think step by step about how to solve this problem.",
            system_prompt="You are a reasoning agent that can use tools to solve problems.",
            max_iterations=5,
        )
        assert config.type == "react"
        assert config.name == "reasoning_agent"
        assert config.description == "An agent that can use tools and reason"
        assert config.tools == ["search", "calculator", "database"]
        assert (
            config.reasoning_prompt
            == "Think step by step about how to solve this problem."
        )
        assert (
            config.system_prompt
            == "You are a reasoning agent that can use tools to solve problems."
        )
        assert config.max_iterations == 5

    def test_minimal_config(self):
        """Test minimal React node configuration."""
        config = ReactNodeConfig(
            type="react",
            name="minimal",
            description="",
            reasoning_prompt=None,
            system_prompt=None,
        )
        assert config.type == "react"
        assert config.name == "minimal"
        assert config.tools == []
        assert config.reasoning_prompt is None
        assert config.system_prompt is None
        assert config.max_iterations == 5

    def test_invalid_max_iterations(self):
        """Test invalid max_iterations."""
        with pytest.raises(ValidationError, match="max_iterations must be positive"):
            ReactNodeConfig(
                type="react",
                name="test",
                max_iterations=0,
                description="",
                reasoning_prompt="",
                system_prompt="",
            )

    def test_empty_tool_names(self):
        """Test empty tool names validation."""
        with pytest.raises(ValidationError, match="tool names cannot be empty"):
            ReactNodeConfig(
                type="react",
                name="test",
                tools=["search", "", "calculator"],
                description="",
                reasoning_prompt="",
                system_prompt="",
            )

    def test_empty_reasoning_prompt(self):
        """Test empty reasoning_prompt validation."""
        with pytest.raises(
            ValidationError, match="reasoning_prompt cannot be empty if provided"
        ):
            ReactNodeConfig(
                type="react",
                name="test",
                reasoning_prompt="",
                description="",
                system_prompt="",
            )

    def test_empty_system_prompt(self):
        """Test empty system_prompt validation."""
        with pytest.raises(
            ValidationError, match="system_prompt cannot be empty if provided"
        ):
            ReactNodeConfig(
                type="react",
                name="test",
                system_prompt="",
                description="",
                reasoning_prompt="",
            )

    def test_tool_stripping(self):
        """Test tool name whitespace stripping."""
        config = ReactNodeConfig(
            type="react",
            name="test",
            tools=["  search  ", "  calculator  "],
            description="",
            reasoning_prompt=None,
            system_prompt=None,
        )
        assert config.tools == ["search", "calculator"]

    def test_prompt_stripping(self):
        """Test prompt whitespace stripping."""
        config = ReactNodeConfig(
            type="react",
            name="test",
            reasoning_prompt="  test reasoning  ",
            system_prompt="  test system  ",
            description="",
        )
        assert config.reasoning_prompt == "test reasoning"
        assert config.system_prompt == "test system"


class TestIntegration:
    """Test integration scenarios."""

    def test_llm_node_creation(self):
        """Test creating LLM node with all fields."""
        config = LLMNodeConfig(
            type="llm",
            name="assistant",
            description="A helpful AI assistant",
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=1000,
            prompt="You are a helpful assistant. Answer the user's question: {user_input}",
            system_prompt="You are a knowledgeable and helpful AI assistant.",
        )

        # Verify all fields are correctly set
        assert config.type == "llm"
        assert config.name == "assistant"
        assert config.description == "A helpful AI assistant"
        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 1000
        if config.prompt is not None:
            assert "user_input" in config.prompt
        if config.system_prompt is not None:
            assert "knowledgeable" in config.system_prompt

    def test_react_node_creation(self):
        """Test creating React node with all fields."""
        config = ReactNodeConfig(
            type="react",
            name="reasoning_agent",
            description="An agent that can use tools and reason",
            tools=["search", "calculator", "database"],
            reasoning_prompt="Think step by step about how to solve this problem.",
            system_prompt="You are a reasoning agent that can use tools to solve problems.",
            max_iterations=5,
        )

        # Verify all fields are correctly set
        assert config.type == "react"
        assert config.name == "reasoning_agent"
        assert config.description == "An agent that can use tools and reason"
        assert len(config.tools) == 3
        assert "search" in config.tools
        assert "calculator" in config.tools
        assert "database" in config.tools
        if config.reasoning_prompt is not None:
            assert "step by step" in config.reasoning_prompt
        if config.system_prompt is not None:
            assert "reasoning agent" in config.system_prompt
        assert config.max_iterations == 5

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test minimum valid values
        llm_config = LLMNodeConfig(
            type="llm",
            name="minimal",
            provider="openai",
            model="gpt-4",
            temperature=0.0,
            max_tokens=1,
            description="",
            prompt=None,
            system_prompt=None,
        )
        assert llm_config.temperature == 0.0
        assert llm_config.max_tokens == 1

        # Test maximum valid values
        llm_config = LLMNodeConfig(
            type="llm",
            name="maximal",
            provider="openai",
            model="gpt-4",
            temperature=2.0,
            max_tokens=8000,
            description="",
            prompt=None,
            system_prompt=None,
        )
        assert llm_config.temperature == 2.0
        assert llm_config.max_tokens == 8000

        # Test React node with minimum iterations
        react_config = ReactNodeConfig(
            type="react",
            name="minimal",
            max_iterations=1,
            description="",
            reasoning_prompt=None,
            system_prompt=None,
        )
        assert react_config.max_iterations == 1
