"""Tests for YAML configuration models."""

from typing import Dict, List

import pytest
from petal.core.config.yaml import (
    BaseNodeConfig,
    LLMNodeConfig,
    ReactNodeConfig,
    StateSchemaConfig,
    TypeResolver,
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
            validate_node_type("unsupported")

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


class TestTypeResolver:
    """Test TypeResolver class."""

    def test_resolve_type_valid_types(self):
        """Test resolving valid type names."""
        assert TypeResolver.resolve_type("str") is str
        assert TypeResolver.resolve_type("int") is int
        assert TypeResolver.resolve_type("float") is float
        assert TypeResolver.resolve_type("bool") is bool
        assert TypeResolver.resolve_type("list") is List
        assert TypeResolver.resolve_type("dict") is Dict
        assert TypeResolver.resolve_type("tuple") is tuple
        assert TypeResolver.resolve_type("set") is set
        assert TypeResolver.resolve_type("bytes") is bytes
        assert TypeResolver.resolve_type("None") is type(None)

    def test_resolve_type_invalid_type(self):
        """Test resolving invalid type name raises error."""
        with pytest.raises(ValueError, match="Unsupported type: invalid_type"):
            TypeResolver.resolve_type("invalid_type")

    def test_is_valid_type(self):
        """Test is_valid_type method."""
        assert TypeResolver.is_valid_type("str") is True
        assert TypeResolver.is_valid_type("int") is True
        assert TypeResolver.is_valid_type("float") is True
        assert TypeResolver.is_valid_type("bool") is True
        assert TypeResolver.is_valid_type("list") is True
        assert TypeResolver.is_valid_type("dict") is True
        assert TypeResolver.is_valid_type("tuple") is True
        assert TypeResolver.is_valid_type("set") is True
        assert TypeResolver.is_valid_type("bytes") is True
        assert TypeResolver.is_valid_type("None") is True
        assert TypeResolver.is_valid_type("invalid_type") is False


class TestStateSchemaConfig:
    """Test StateSchemaConfig model."""

    def test_valid_config(self):
        """Test valid state schema configuration."""
        config = StateSchemaConfig(
            fields={"input": "str", "output": "str"}, required_fields=["input"]
        )
        assert config.fields == {"input": "str", "output": "str"}
        assert config.required_fields == ["input"]

    def test_default_config(self):
        """Test default state schema configuration."""
        config = StateSchemaConfig()
        assert config.fields == {}
        assert config.required_fields == []

    def test_get_resolved_fields(self):
        """Test get_resolved_fields method."""
        config = StateSchemaConfig(
            fields={"user_input": "str", "confidence": "float", "metadata": "dict"}
        )
        resolved_fields = config.get_resolved_fields()
        assert resolved_fields["user_input"] is str
        assert resolved_fields["confidence"] is float
        assert resolved_fields["metadata"] is Dict

    def test_get_resolved_fields_empty(self):
        """Test get_resolved_fields with empty fields."""
        config = StateSchemaConfig()
        resolved_fields = config.get_resolved_fields()
        assert resolved_fields == {}

    def test_get_resolved_fields_complex_types(self):
        """Test get_resolved_fields with complex types."""
        config = StateSchemaConfig(
            fields={
                "text": "str",
                "numbers": "list",
                "settings": "dict",
                "enabled": "bool",
                "count": "int",
                "score": "float",
                "data": "bytes",
                "items": "tuple",
                "tags": "set",
                "none_value": "None",
            }
        )
        resolved_fields = config.get_resolved_fields()
        assert resolved_fields["text"] is str
        assert resolved_fields["numbers"] is List
        assert resolved_fields["settings"] is Dict
        assert resolved_fields["enabled"] is bool
        assert resolved_fields["count"] is int
        assert resolved_fields["score"] is float
        assert resolved_fields["data"] is bytes
        assert resolved_fields["items"] is tuple
        assert resolved_fields["tags"] is set
        assert resolved_fields["none_value"] is type(None)


class TestValidationConfig:
    """Test ValidationConfig model."""

    def test_valid_config(self):
        """Test valid validation configuration."""
        input_schema = StateSchemaConfig(fields={"input": "str"})
        output_schema = StateSchemaConfig(fields={"output": "str"})

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

    def test_base_node_config_creation(self):
        """Test BaseNodeConfig creation with required fields."""
        config = BaseNodeConfig(
            type="llm",
            name="test_node",
            description="Test node",
            enabled=True,
            state_schema=None,
            input_schema=None,
            output_schema=None,
        )
        assert config.type == "llm"
        assert config.name == "test_node"
        assert config.description == "Test node"
        assert config.enabled is True

    def test_base_node_config_minimal(self):
        """Test BaseNodeConfig creation with minimal fields."""
        config = BaseNodeConfig(
            type="react",
            name="minimal_node",
            state_schema=None,
            input_schema=None,
            output_schema=None,
        )
        assert config.type == "react"
        assert config.name == "minimal_node"
        assert config.description is None
        assert config.enabled is True

    def test_base_node_config_with_schemas(self):
        """Test BaseNodeConfig creation with schema configurations."""
        state_schema = StateSchemaConfig(
            fields={"user_input": "str", "confidence": "float"},
            required_fields=["user_input"],
            optional_fields={"confidence": 0.5},
        )
        input_schema = StateSchemaConfig(fields={"text": "str"})
        output_schema = StateSchemaConfig(fields={"result": "str"})
        config = BaseNodeConfig(
            type="llm",
            name="schema_node",
            state_schema=state_schema,
            input_schema=input_schema,
            output_schema=output_schema,
        )
        assert config.state_schema is not None
        assert config.input_schema is not None
        assert config.output_schema is not None

    def test_base_node_config_validation(self):
        """Test BaseNodeConfig field validation."""
        # Test empty name
        with pytest.raises(ValueError, match="name cannot be empty"):
            BaseNodeConfig(
                type="llm",
                name="",
                state_schema=None,
                input_schema=None,
                output_schema=None,
            )

        # Test whitespace name
        with pytest.raises(ValueError, match="name cannot be empty"):
            BaseNodeConfig(
                type="llm",
                name="   ",
                state_schema=None,
                input_schema=None,
                output_schema=None,
            )

        # Test invalid node type
        with pytest.raises(ValueError, match="node_type must be one of"):
            BaseNodeConfig(
                type="invalid",
                name="test",
                state_schema=None,
                input_schema=None,
                output_schema=None,
            )


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
            tools=[],
            description="",
            reasoning_prompt=None,
            system_prompt=None,
        )
        assert config.type == "react"
        assert config.name == "minimal"
        assert config.tools == []
        assert config.max_iterations == 5
        assert config.reasoning_prompt is None
        assert config.system_prompt is None

    def test_invalid_max_iterations(self):
        """Test invalid max_iterations."""
        with pytest.raises(ValidationError, match="max_iterations must be positive"):
            ReactNodeConfig(
                type="react",
                name="test",
                tools=["tool1"],
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
                tools=["tool1", "", "tool3"],
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
                tools=["tool1"],
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
                tools=["tool1"],
                system_prompt="",
                description="",
                reasoning_prompt="",
            )

    def test_tool_stripping(self):
        """Test tool name whitespace stripping."""
        config = ReactNodeConfig(
            type="react",
            name="test",
            tools=["  tool1  ", "  tool2  "],
            description="",
            reasoning_prompt=None,
            system_prompt=None,
        )
        assert config.tools == ["tool1", "tool2"]

    def test_prompt_stripping(self):
        """Test prompt whitespace stripping."""
        config = ReactNodeConfig(
            type="react",
            name="test",
            tools=["tool1"],
            reasoning_prompt="  test reasoning  ",
            system_prompt="  test system  ",
            description="",
        )
        assert config.reasoning_prompt == "test reasoning"
        assert config.system_prompt == "test system"


class TestIntegration:
    """Test integration between different config models."""

    def test_llm_node_creation(self):
        """Test LLM node creation with all components."""
        state_schema = StateSchemaConfig(
            fields={"user_input": "str", "response": "str"},
            required_fields=["user_input"],
        )

        config = LLMNodeConfig(
            type="llm",
            name="assistant",
            description="A helpful assistant",
            provider="openai",
            model="gpt-4o-mini",
            state_schema=state_schema,
        )

        assert config.state_schema == state_schema
        assert config.state_schema.fields == {"user_input": "str", "response": "str"}

    def test_react_node_creation(self):
        """Test React node creation with all components."""
        state_schema = StateSchemaConfig(
            fields={"query": "str", "tools_used": "list", "result": "str"},
            required_fields=["query"],
        )

        config = ReactNodeConfig(
            type="react",
            name="reasoning_agent",
            description="An agent that can use tools and reason",
            tools=["search", "calculator"],
            state_schema=state_schema,
        )

        assert config.state_schema == state_schema
        assert config.state_schema.fields == {
            "query": "str",
            "tools_used": "list",
            "result": "str",
        }

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with None state schema
        llm_config = LLMNodeConfig(
            type="llm",
            name="test",
            provider="openai",
            model="gpt-4",
            state_schema=None,
        )
        assert llm_config.state_schema is None

        # Test with empty state schema
        empty_schema = StateSchemaConfig()
        react_config = ReactNodeConfig(
            type="react",
            name="test",
            tools=[],
            state_schema=empty_schema,
        )
        assert react_config.state_schema == empty_schema


class TestEnhancedStateSchemaConfig:
    """Test enhanced StateSchemaConfig model with string type support."""

    def test_string_type_fields(self):
        """Test state schema with string type definitions."""
        config = StateSchemaConfig(
            fields={"user_input": "str", "confidence": "float", "metadata": "dict"},
            required_fields=["user_input"],
        )
        assert config.fields == {
            "user_input": "str",
            "confidence": "float",
            "metadata": "dict",
        }
        assert config.required_fields == ["user_input"]

    def test_optional_fields_with_defaults(self):
        """Test state schema with optional fields and default values."""
        config = StateSchemaConfig(
            fields={"user_input": "str", "confidence": "float"},
            required_fields=["user_input"],
            optional_fields={"confidence": 0.5},
        )
        assert config.optional_fields == {"confidence": 0.5}

    def test_nested_schemas(self):
        """Test state schema with nested schema definitions."""
        nested_schema = StateSchemaConfig(
            fields={"algorithm": "str", "parameters": "dict"},
            required_fields=["algorithm"],
        )
        config = StateSchemaConfig(
            fields={"data_input": "dict", "processing_config": "dict"},
            required_fields=["data_input"],
            nested_schemas={"processing_config": nested_schema},
        )
        assert "processing_config" in config.nested_schemas
        assert config.nested_schemas["processing_config"] == nested_schema

    def test_invalid_field_name(self):
        """Test validation of invalid field names."""
        with pytest.raises(ValidationError, match="Invalid field name"):
            StateSchemaConfig(
                fields={"invalid-field": "str"}, required_fields=["invalid-field"]
            )

    def test_invalid_type_name(self):
        """Test validation of invalid type names."""
        with pytest.raises(ValidationError, match="Unsupported type"):
            StateSchemaConfig(
                fields={"user_input": "invalid_type"}, required_fields=["user_input"]
            )

    def test_required_field_not_in_fields(self):
        """Test validation when required field is not in fields."""
        with pytest.raises(
            ValidationError, match="Required field not found in fields: missing_field"
        ):
            StateSchemaConfig(
                fields={"user_input": "str"}, required_fields=["missing_field"]
            )

    def test_optional_field_not_in_fields(self):
        """Test validation when optional field is not in fields."""
        with pytest.raises(
            ValidationError, match="Optional field not found in fields: missing_field"
        ):
            StateSchemaConfig(
                fields={"user_input": "str"},
                optional_fields={"missing_field": "default_value"},
            )
