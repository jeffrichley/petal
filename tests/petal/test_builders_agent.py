"""Unit tests for AgentBuilder."""

import pytest
from petal.core.builders.agent import AgentBuilder
from petal.core.config.agent import AgentConfig
from petal.core.config.checkpointer import CheckpointerConfig
from petal.core.steps.registry import StepRegistry
from typing_extensions import TypedDict


class BuilderTestState(TypedDict):
    """Test state type."""

    messages: list
    processed: bool


class TestAgentBuilder:
    """Test cases for AgentBuilder."""

    def test_agent_builder_initialization(self):
        """Test that AgentBuilder initializes correctly."""
        builder = AgentBuilder(BuilderTestState)

        assert builder._config is not None
        assert isinstance(builder._config, AgentConfig)
        assert builder._config.state_type == BuilderTestState
        assert builder._registry is not None
        assert isinstance(builder._registry, StepRegistry)

    def test_with_step_fluent_interface(self):
        """Test that with_step returns self for fluent chaining."""
        builder = AgentBuilder(BuilderTestState)

        result = builder.with_step("llm", prompt_template="Hello {name}")

        assert result is builder
        assert len(builder._config.steps) == 1
        assert builder._config.steps[0].strategy_type == "llm"
        assert builder._config.steps[0].config["prompt_template"] == "Hello {name}"

    def test_with_step_custom_node_name(self):
        """Test that with_step accepts custom node names."""
        builder = AgentBuilder(BuilderTestState)

        builder.with_step("llm", prompt_template="Hello", node_name="custom_node")

        assert len(builder._config.steps) == 1
        assert builder._config.steps[0].node_name == "custom_node"

    def test_with_memory_fluent_interface(self):
        """Test that with_memory returns self for fluent chaining."""
        builder = AgentBuilder(BuilderTestState)

        memory_config = {"memory_type": "conversation", "max_tokens": 1000}
        result = builder.with_memory(memory_config)

        assert result is builder
        assert builder._config.memory is not None
        assert builder._config.memory.memory_type == "conversation"
        assert builder._config.memory.max_tokens == 1000

    def test_with_memory_pydantic_validation(self):
        """Test that with_memory uses Pydantic validation."""
        builder = AgentBuilder(BuilderTestState)

        # Valid memory config
        memory_config = {"memory_type": "conversation", "max_tokens": 500}
        builder.with_memory(memory_config)

        assert builder._config.memory is not None
        assert builder._config.memory.memory_type == "conversation"
        assert builder._config.memory.max_tokens == 500

    def test_with_memory_invalid_config_raises_error(self):
        """Test that invalid memory config raises validation error."""
        builder = AgentBuilder(BuilderTestState)

        # Invalid memory type
        with pytest.raises(ValueError, match="memory_type must be one of"):
            builder.with_memory({"memory_type": "invalid_type"})

    def test_with_llm_fluent_interface(self):
        """Test that with_llm returns self for fluent chaining."""
        builder = AgentBuilder(BuilderTestState)

        result = builder.with_llm(
            provider="openai", model="gpt-4o-mini", temperature=0.1
        )

        assert result is builder
        assert builder._config.llm_config is not None
        assert builder._config.llm_config.provider == "openai"
        assert builder._config.llm_config.model == "gpt-4o-mini"
        assert builder._config.llm_config.temperature == 0.1

    def test_with_llm_pydantic_validation(self):
        """Test that with_llm uses Pydantic validation."""
        builder = AgentBuilder(BuilderTestState)

        # Valid LLM config
        builder.with_llm(provider="openai", model="gpt-4o-mini")

        assert builder._config.llm_config is not None
        assert builder._config.llm_config.provider == "openai"
        assert builder._config.llm_config.model == "gpt-4o-mini"

    def test_with_llm_invalid_config_raises_error(self):
        """Test that invalid LLM config raises validation error."""
        builder = AgentBuilder(BuilderTestState)

        # Invalid provider
        with pytest.raises(ValueError, match="provider must be one of"):
            builder.with_llm(provider="invalid_provider", model="test")

    def test_with_logging_fluent_interface(self):
        """Test that with_logging returns self for fluent chaining."""
        builder = AgentBuilder(BuilderTestState)

        logging_config = {"enabled": True, "level": "DEBUG", "include_state": True}
        result = builder.with_logging(logging_config)

        assert result is builder
        assert builder._config.logging_config is not None
        assert builder._config.logging_config.enabled is True
        assert builder._config.logging_config.level == "DEBUG"
        assert builder._config.logging_config.include_state is True

    def test_with_graph_config_fluent_interface(self):
        """Test that with_graph_config returns self for fluent chaining."""
        builder = AgentBuilder(BuilderTestState)

        graph_config = {"graph_type": "linear", "allow_parallel": True}
        result = builder.with_graph_config(graph_config)

        assert result is builder
        assert builder._config.graph_config.graph_type == "linear"
        assert builder._config.graph_config.allow_parallel is True

    def test_with_checkpointer_fluent_interface(self):
        """Test that with_checkpointer returns self for fluent chaining."""
        builder = AgentBuilder(BuilderTestState)

        checkpointer_config = CheckpointerConfig(type="memory", enabled=True)
        result = builder.with_checkpointer(checkpointer_config)

        assert result is builder
        assert builder._config.checkpointer is not None
        assert builder._config.checkpointer.type == "memory"
        assert builder._config.checkpointer.enabled is True

    def test_with_checkpointer_pydantic_validation(self):
        """Test that with_checkpointer uses Pydantic validation."""
        builder = AgentBuilder(BuilderTestState)

        # Valid checkpointer config
        checkpointer_config = CheckpointerConfig(type="memory", enabled=True)
        builder.with_checkpointer(checkpointer_config)

        assert builder._config.checkpointer is not None
        assert builder._config.checkpointer.type == "memory"
        assert builder._config.checkpointer.enabled is True

    def test_with_checkpointer_invalid_config_raises_error(self):
        """Test that invalid checkpointer config raises validation error."""
        builder = AgentBuilder(BuilderTestState)

        # Invalid checkpointer type - this will raise a Pydantic validation error
        with pytest.raises(
            ValueError, match="Input should be 'memory', 'postgres' or 'sqlite'"
        ):
            checkpointer_config = CheckpointerConfig(type="invalid_type", enabled=True)  # type: ignore[arg-type]
            builder.with_checkpointer(checkpointer_config)

    def test_method_chaining(self):
        """Test that multiple methods can be chained together."""
        builder = AgentBuilder(BuilderTestState)

        result = (
            builder.with_step("llm", prompt_template="Hello {name}")
            .with_memory({"memory_type": "conversation"})
            .with_llm(provider="openai", model="gpt-4o-mini")
            .with_logging({"enabled": True})
            .with_graph_config({"graph_type": "linear"})
        )

        assert result is builder
        assert len(builder._config.steps) == 1
        assert builder._config.memory is not None
        assert builder._config.llm_config is not None
        assert builder._config.logging_config is not None
        assert builder._config.graph_config is not None

    def test_with_step_unknown_strategy_raises_error(self):
        """Test that unknown step strategy raises error."""
        builder = AgentBuilder(BuilderTestState)

        with pytest.raises(ValueError, match="Unknown step type"):
            builder.with_step("unknown_strategy", prompt_template="Hello")

    @pytest.mark.asyncio
    async def test_build_method_validation(self):
        """Test that build method validates configuration properly."""
        builder = AgentBuilder(BuilderTestState)

        # Build should raise ValueError when no steps are configured
        with pytest.raises(ValueError, match="Cannot build agent: no steps configured"):
            await builder.build()

    def test_builder_state_consistency(self):
        """Test that builder maintains consistent state."""
        builder = AgentBuilder(BuilderTestState)

        # Add multiple steps
        builder.with_step("llm", prompt_template="Step 1")
        builder.with_step("llm", prompt_template="Step 2")

        assert len(builder._config.steps) == 2
        assert builder._config.steps[0].config["prompt_template"] == "Step 1"
        assert builder._config.steps[1].config["prompt_template"] == "Step 2"

    def test_builder_with_different_state_types(self):
        """Test that builder works with different state types."""

        class CustomState(TypedDict):
            custom_field: str

        builder = AgentBuilder(CustomState)

        assert builder._config.state_type == CustomState
        assert len(builder._config.steps) == 0

    def test_agent_builder_init_with_none_state_type(self):
        """Test that AgentBuilder raises error when state_type is None."""
        with pytest.raises(ValueError, match="state_type cannot be None"):
            AgentBuilder(None)  # type: ignore[arg-type]

    def test_builder_configuration_validation(self):
        """Test that builder validates configuration properly."""
        builder = AgentBuilder(BuilderTestState)

        # Test that invalid configurations are caught
        with pytest.raises(ValueError):
            builder.with_memory({"memory_type": "invalid"})

        with pytest.raises(ValueError):
            builder.with_llm(provider="invalid", model="test")

        with pytest.raises(ValueError):
            builder.with_logging({"level": "INVALID"})

        with pytest.raises(ValueError):
            builder.with_graph_config({"graph_type": "invalid"})

    def test_get_config_method(self):
        """Test that get_config returns the current configuration."""
        builder = AgentBuilder(BuilderTestState)

        config = builder.get_config()

        assert config is builder._config
        assert isinstance(config, AgentConfig)
        assert config.state_type == BuilderTestState

    def test_get_registry_method(self):
        """Test that get_registry returns the step registry."""
        builder = AgentBuilder(BuilderTestState)

        registry = builder.get_registry()

        assert registry is builder._registry
        assert isinstance(registry, StepRegistry)

    def test_with_logging_success_path(self):
        """Test that with_logging succeeds with valid configuration."""
        builder = AgentBuilder(BuilderTestState)

        # Valid logging config
        logging_config = {"enabled": True, "level": "INFO"}
        result = builder.with_logging(logging_config)

        assert result is builder
        assert builder._config.logging_config is not None
        assert builder._config.logging_config.enabled is True
        assert builder._config.logging_config.level == "INFO"

    def test_with_graph_config_success_path(self):
        """Test that with_graph_config succeeds with valid configuration."""
        builder = AgentBuilder(BuilderTestState)

        # Valid graph config
        graph_config = {"graph_type": "linear", "allow_parallel": False}
        result = builder.with_graph_config(graph_config)

        assert result is builder
        assert builder._config.graph_config is not None
        assert builder._config.graph_config.graph_type == "linear"
        assert builder._config.graph_config.allow_parallel is False

    def test_with_system_prompt_adds_prompt_to_llm_step(self):
        """Test that with_system_prompt adds the prompt to the most recent LLM step."""
        builder = AgentBuilder(BuilderTestState)
        builder.with_step("llm", prompt_template="Hello")
        result = builder.with_system_prompt("System prompt here")
        assert result is builder
        assert builder._config.steps[-1].config["system_prompt"] == "System prompt here"

    def test_with_system_prompt_raises_if_no_steps(self):
        """Test that with_system_prompt raises ValueError if no steps exist."""
        builder = AgentBuilder(BuilderTestState)
        with pytest.raises(ValueError, match="no steps have been added"):
            builder.with_system_prompt("System prompt")

    def test_with_system_prompt_raises_if_not_llm(self):
        """Test that with_system_prompt raises ValueError if last step is not LLM."""
        builder = AgentBuilder(BuilderTestState)
        builder.with_step("custom", some_param=123)
        with pytest.raises(ValueError, match="most recent step is 'custom', not 'llm'"):
            builder.with_system_prompt("System prompt")
