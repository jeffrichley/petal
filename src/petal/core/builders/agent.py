"""AgentBuilder implementation with fluent interface."""

from typing import Any, Dict, Optional, Type

from petal.core.builders.director import AgentBuilderDirector
from petal.core.config.agent import (
    AgentConfig,
    GraphConfig,
    LLMConfig,
    LoggingConfig,
    MemoryConfig,
    StepConfig,
)
from petal.core.steps.registry import StepRegistry


class AgentBuilder:
    """
    Fluent interface for building agents using composition with AgentConfig and StepRegistry.

    This class provides a builder pattern implementation that uses composition
    with AgentConfig for configuration management and StepRegistry for step
    strategy resolution. It provides a fluent interface for easy agent construction.
    """

    def __init__(self, state_type: Type):
        """
        Initialize the AgentBuilder.

        Args:
            state_type: The type for agent state (e.g., TypedDict class)
        """
        if state_type is None:
            raise ValueError("state_type cannot be None")

        self._config = AgentConfig(
            name=None,
            state_type=state_type,
            steps=[],
            memory=None,
            graph_config=GraphConfig(),
            llm_config=None,
            logging_config=LoggingConfig(),
        )
        self._registry = StepRegistry()

    def with_step(
        self, step_type: str, node_name: Optional[str] = None, **config: Any
    ) -> "AgentBuilder":
        """
        Add a step to the agent configuration.

        Args:
            step_type: The type of step strategy to use (e.g., "llm", "custom")
            node_name: Optional custom node name for the step
            **config: Configuration parameters for the step

        Returns:
            self: For method chaining

        Raises:
            ValueError: If the step type is not registered in the registry
        """
        # Validate that the step_type is registered
        self._registry.validate_strategy(step_type)

        # Create step configuration
        step_config = StepConfig(
            strategy_type=step_type, config=config, node_name=node_name
        )

        # Add step to configuration
        self._config.add_step(step_config)

        return self

    def with_memory(self, memory_config: Dict[str, Any]) -> "AgentBuilder":
        """
        Add memory configuration to the agent.

        Args:
            memory_config: Dictionary containing memory configuration parameters

        Returns:
            self: For method chaining

        Raises:
            ValueError: If memory configuration is invalid
        """
        try:
            memory = MemoryConfig(**memory_config)
            self._config.set_memory(memory)
        except Exception as e:
            raise ValueError(f"Invalid memory configuration: {e}") from e

        return self

    def with_llm(self, llm_config: Dict[str, Any]) -> "AgentBuilder":
        """
        Add LLM configuration to the agent.

        Args:
            llm_config: Dictionary containing LLM configuration parameters

        Returns:
            self: For method chaining

        Raises:
            ValueError: If LLM configuration is invalid
        """
        try:
            llm = LLMConfig(**llm_config)
            self._config.set_llm(llm)
        except Exception as e:
            raise ValueError(f"Invalid LLM configuration: {e}") from e

        return self

    def with_logging(self, logging_config: Dict[str, Any]) -> "AgentBuilder":
        """
        Add logging configuration to the agent.

        Args:
            logging_config: Dictionary containing logging configuration parameters

        Returns:
            self: For method chaining

        Raises:
            ValueError: If logging configuration is invalid
        """
        try:
            logging = LoggingConfig(**logging_config)
            self._config.set_logging(logging)
        except Exception as e:
            raise ValueError(f"Invalid logging configuration: {e}") from e

        return self

    def with_graph_config(self, graph_config: Dict[str, Any]) -> "AgentBuilder":
        """
        Add graph configuration to the agent.

        Args:
            graph_config: Dictionary containing graph configuration parameters

        Returns:
            self: For method chaining

        Raises:
            ValueError: If graph configuration is invalid
        """
        try:
            graph = GraphConfig(**graph_config)
            self._config.graph_config = graph
        except Exception as e:
            raise ValueError(f"Invalid graph configuration: {e}") from e

        return self

    def build(self) -> Any:
        """
        Build the agent from configuration using AgentBuilderDirector (MCP-compliant).

        Returns:
            The built agent
        """
        director = AgentBuilderDirector(self._config, self._registry)
        return director.build()

    def get_config(self) -> AgentConfig:
        """
        Get the current configuration.

        Returns:
            The current AgentConfig instance
        """
        return self._config

    def get_registry(self) -> StepRegistry:
        """
        Get the step registry.

        Returns:
            The current StepRegistry instance
        """
        return self._registry
