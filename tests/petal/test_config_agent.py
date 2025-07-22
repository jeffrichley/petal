"""Tests for AgentConfig and related configuration classes."""

import pytest

from petal.core.config.agent import (
    AgentConfig,
    GraphConfig,
    LLMConfig,
    LoggingConfig,
    MemoryConfig,
    StepConfig,
)


class TestStepConfig:
    """Test StepConfig Pydantic model."""

    def test_step_config_creation(self):
        """Test basic StepConfig creation."""
        config = StepConfig(
            strategy_type="llm",
            config={"prompt_template": "Hello {name}", "model": "gpt-4o-mini"},
            node_name=None,
        )
        assert config.strategy_type == "llm"
        assert config.config["prompt_template"] == "Hello {name}"
        assert config.config["model"] == "gpt-4o-mini"

    def test_step_config_validation(self):
        """Test StepConfig validation."""
        # Valid config
        config = StepConfig(
            strategy_type="llm", config={"prompt_template": "Hello"}, node_name=None
        )
        assert config.strategy_type == "llm"

        # Invalid config - missing required fields
        with pytest.raises(ValueError):
            StepConfig(strategy_type="", config={}, node_name=None)

    def test_step_config_optional_fields(self):
        """Test StepConfig with optional fields."""
        config = StepConfig(
            strategy_type="llm",
            config={"prompt_template": "Hello"},
            node_name="custom_node",
        )
        assert config.node_name == "custom_node"


class TestMemoryConfig:
    """Test MemoryConfig Pydantic model."""

    def test_memory_config_creation(self):
        """Test basic MemoryConfig creation."""
        config = MemoryConfig(memory_type="conversation", max_tokens=1000)
        assert config.memory_type == "conversation"
        assert config.max_tokens == 1000

    def test_memory_config_defaults(self):
        """Test MemoryConfig with defaults."""
        config = MemoryConfig(memory_type="conversation")
        assert config.memory_type == "conversation"
        assert config.max_tokens == 500  # default value
        assert config.enabled is True  # default value

    def test_memory_config_validation(self):
        """Test MemoryConfig validation."""
        # Valid config
        config = MemoryConfig(memory_type="conversation")
        assert config.memory_type == "conversation"

        # Invalid memory type
        with pytest.raises(ValueError):
            MemoryConfig(memory_type="invalid_type")

        # Invalid max_tokens
        with pytest.raises(ValueError):
            MemoryConfig(memory_type="conversation", max_tokens=-1)


class TestGraphConfig:
    """Test GraphConfig Pydantic model."""

    def test_graph_config_creation(self):
        """Test basic GraphConfig creation."""
        config = GraphConfig(graph_type="linear", allow_parallel=False)
        assert config.graph_type == "linear"
        assert config.allow_parallel is False

    def test_graph_config_defaults(self):
        """Test GraphConfig with defaults."""
        config = GraphConfig()
        assert config.graph_type == "linear"  # default
        assert config.allow_parallel is False  # default
        assert config.max_retries == 3  # default

    def test_graph_config_validation(self):
        """Test GraphConfig validation."""
        # Valid config
        config = GraphConfig(graph_type="linear")
        assert config.graph_type == "linear"

        # Invalid graph type
        with pytest.raises(ValueError):
            GraphConfig(graph_type="invalid_type")

        # Invalid max_retries
        with pytest.raises(ValueError):
            GraphConfig(max_retries=-1)


class TestLLMConfig:
    """Test LLMConfig Pydantic model."""

    def test_llm_config_creation(self):
        """Test basic LLMConfig creation."""
        config = LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.7)
        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.7

    def test_llm_config_defaults(self):
        """Test LLMConfig with defaults."""
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0  # default
        assert config.max_tokens == 1000  # default

    def test_llm_config_validation(self):
        """Test LLMConfig validation."""
        # Valid config
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        assert config.provider == "openai"

        # Invalid provider
        with pytest.raises(ValueError):
            LLMConfig(provider="invalid_provider", model="gpt-4o-mini")

        # Invalid temperature
        with pytest.raises(ValueError):
            LLMConfig(provider="openai", model="gpt-4o-mini", temperature=2.5)

        # Invalid max_tokens
        with pytest.raises(ValueError):
            LLMConfig(provider="openai", model="gpt-4o-mini", max_tokens=-1)

    def test_llm_config_temperature_below_range(self):
        """Test that LLMConfig raises error for temperature below 0.0."""
        with pytest.raises(
            ValueError, match="Input should be greater than or equal to 0"
        ):
            LLMConfig(provider="openai", model="gpt-4o", temperature=-0.1)

    def test_llm_config_temperature_above_range(self):
        """Test that LLMConfig raises error for temperature above 2.0."""
        with pytest.raises(ValueError, match="Input should be less than or equal to 2"):
            LLMConfig(provider="openai", model="gpt-4o", temperature=2.1)

    def test_llm_config_max_tokens_below_one(self):
        """Test that LLMConfig raises error for max_tokens below 1."""
        with pytest.raises(
            ValueError, match="Input should be greater than or equal to 1"
        ):
            LLMConfig(provider="openai", model="gpt-4o", max_tokens=0)


class TestLoggingConfig:
    """Test LoggingConfig Pydantic model."""

    def test_logging_config_creation(self):
        """Test basic LoggingConfig creation."""
        config = LoggingConfig(enabled=True, level="INFO", include_state=True)
        assert config.enabled is True
        assert config.level == "INFO"
        assert config.include_state is True

    def test_logging_config_defaults(self):
        """Test LoggingConfig with defaults."""
        config = LoggingConfig()
        assert config.enabled is True  # default
        assert config.level == "INFO"  # default
        assert config.include_state is False  # default

    def test_logging_config_validation(self):
        """Test LoggingConfig validation."""
        # Valid config
        config = LoggingConfig(level="DEBUG")
        assert config.level == "DEBUG"

        # Invalid level
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID_LEVEL")


class TestAgentConfig:
    """Test AgentConfig Pydantic model."""

    def test_agent_config_creation(self):
        """Test basic AgentConfig creation."""
        config = AgentConfig(
            name="test_agent", state_type=dict, memory=None, llm_config=None
        )
        assert config.name == "test_agent"
        assert config.state_type is dict
        assert len(config.steps) == 0  # empty list by default

    def test_agent_config_defaults(self):
        """Test AgentConfig with defaults."""
        config = AgentConfig(state_type=dict, name=None, memory=None, llm_config=None)
        assert config.name is None  # default
        assert config.state_type is dict
        assert config.memory is None  # default
        assert config.graph_config is not None  # default empty config
        assert config.llm_config is None  # default
        assert config.logging_config is not None  # default config

    def test_agent_config_add_step(self):
        """Test adding steps to AgentConfig."""
        config = AgentConfig(state_type=dict, name=None, memory=None, llm_config=None)

        step_config = StepConfig(
            strategy_type="llm",
            config={"prompt_template": "Hello {name}"},
            node_name=None,
        )

        config.add_step(step_config)
        assert len(config.steps) == 1
        assert config.steps[0].strategy_type == "llm"
        assert config.steps[0].config["prompt_template"] == "Hello {name}"

    def test_agent_config_set_memory(self):
        """Test setting memory configuration."""
        config = AgentConfig(state_type=dict, name=None, memory=None, llm_config=None)

        memory_config = MemoryConfig(memory_type="conversation", max_tokens=1000)

        config.set_memory(memory_config)
        assert config.memory is not None
        assert config.memory.memory_type == "conversation"
        assert config.memory.max_tokens == 1000

    def test_agent_config_set_llm(self):
        """Test setting LLM configuration."""
        config = AgentConfig(state_type=dict, name=None, memory=None, llm_config=None)

        llm_config = LLMConfig(provider="openai", model="gpt-4o-mini")

        config.set_llm(llm_config)
        assert config.llm_config is not None
        assert config.llm_config.provider == "openai"
        assert config.llm_config.model == "gpt-4o-mini"

    def test_agent_config_set_logging(self):
        """Test setting logging configuration."""
        config = AgentConfig(state_type=dict, name=None, memory=None, llm_config=None)

        logging_config = LoggingConfig(enabled=True, level="DEBUG")

        config.set_logging(logging_config)
        assert config.logging_config.enabled is True
        assert config.logging_config.level == "DEBUG"

    def test_agent_config_validation(self):
        """Test AgentConfig validation."""
        # Valid config - Pydantic validation happens automatically
        _ = AgentConfig(
            state_type=dict, name=None, memory=None, llm_config=None
        )  # Test that creation works without errors
        # No need to call validate() - Pydantic handles it automatically

    def test_agent_config_validation_with_empty_steps(self):
        """Test that AgentConfig validation works with empty steps."""
        config = AgentConfig(state_type=dict, name=None, memory=None, llm_config=None)
        # Should not raise any validation errors
        assert config.steps == []

    def test_agent_config_to_dict(self):
        """Test that AgentConfig can be converted to dictionary."""
        config = AgentConfig(
            state_type=dict, name="test_agent", memory=None, llm_config=None
        )
        config.add_step(
            StepConfig(strategy_type="llm", config={"prompt": "test"}, node_name=None)
        )

        config_dict = config.to_dict()

        assert config_dict["name"] == "test_agent"
        assert config_dict["state_type"] == "dict"
        assert len(config_dict["steps"]) == 1
        assert config_dict["steps"][0]["strategy_type"] == "llm"

    def test_agent_config_from_dict(self):
        """Test that AgentConfig can be created from dictionary."""
        data = {
            "name": "test_agent",
            "state_type": "dict",
            "steps": [
                {
                    "strategy_type": "llm",
                    "config": {"prompt": "test"},
                    "node_name": None,
                }
            ],
        }

        config = AgentConfig.from_dict(data)

        assert config.name == "test_agent"
        assert config.state_type is dict  # Should default to dict
        assert len(config.steps) == 1
        assert config.steps[0].strategy_type == "llm"

    def test_agent_config_copy(self):
        """Test copying AgentConfig using Pydantic's built-in model_copy."""
        config = AgentConfig(
            name="test_agent", state_type=dict, memory=None, llm_config=None
        )

        step_config = StepConfig(
            strategy_type="llm", config={"prompt_template": "Hello"}, node_name=None
        )
        config.add_step(step_config)

        copied_config = config.model_copy()
        assert copied_config.name == config.name
        assert len(copied_config.steps) == len(config.steps)
        assert copied_config is not config  # Different objects

    def test_agent_config_mutability(self):
        """Test that AgentConfig fields are properly mutable where appropriate."""
        config = AgentConfig(state_type=dict, name=None, memory=None, llm_config=None)

        # Steps list should be mutable (we need to add steps)
        assert isinstance(config.steps, list)

        # Graph config should be mutable
        assert isinstance(config.graph_config, GraphConfig)

        # Pydantic models are mutable by default, so we can modify them
        # This test verifies the actual behavior rather than assuming immutability
        config.graph_config.graph_type = "branching"
        assert config.graph_config.graph_type == "branching"


class TestAgentConfigIntegration:
    """Test AgentConfig integration scenarios."""

    def test_agent_config_with_all_components(self):
        """Test AgentConfig with all components configured."""
        config = AgentConfig(
            name="full_agent", state_type=dict, memory=None, llm_config=None
        )

        # Add steps
        step1 = StepConfig(
            strategy_type="llm",
            config={"prompt_template": "Step 1: {input}"},
            node_name=None,
        )
        step2 = StepConfig(
            strategy_type="custom", config={"function": "process_data"}, node_name=None
        )

        config.add_step(step1)
        config.add_step(step2)

        # Set memory
        memory_config = MemoryConfig(memory_type="conversation", max_tokens=2000)
        config.set_memory(memory_config)

        # Set LLM
        llm_config = LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.5)
        config.set_llm(llm_config)

        # Set logging
        logging_config = LoggingConfig(enabled=True, level="DEBUG", include_state=True)
        config.set_logging(logging_config)

        # Validate - Pydantic validation happens automatically
        # No need to call validate() explicitly

        # Assertions
        assert config.name == "full_agent"
        assert len(config.steps) == 2
        assert config.steps[0].strategy_type == "llm"
        assert config.steps[1].strategy_type == "custom"
        assert config.memory is not None
        assert config.memory.memory_type == "conversation"
        assert config.llm_config is not None
        assert config.llm_config.provider == "openai"
        assert config.logging_config.level == "DEBUG"

    def test_agent_config_serialization_roundtrip(self):
        """Test that AgentConfig can be serialized and deserialized."""
        original_config = AgentConfig(
            name="serialization_test", state_type=dict, memory=None, llm_config=None
        )

        step_config = StepConfig(
            strategy_type="llm",
            config={"prompt_template": "Hello {name}"},
            node_name=None,
        )
        original_config.add_step(step_config)

        # Convert to dict and back
        config_dict = original_config.to_dict()
        restored_config = AgentConfig.from_dict(config_dict)

        # Should be equivalent
        assert restored_config.name == original_config.name
        assert len(restored_config.steps) == len(original_config.steps)
        assert (
            restored_config.steps[0].strategy_type
            == original_config.steps[0].strategy_type
        )
