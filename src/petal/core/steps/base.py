"""Base classes for step creation strategies."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict


class StepStrategy(ABC):
    """Abstract base class for step creation strategies."""

    @abstractmethod
    async def create_step(self, config: Dict[str, Any]) -> Callable:
        """Create a step callable from configuration."""

    @abstractmethod
    def get_node_name(self, index: int) -> str:
        """Generate a node name for the step at the given index."""


class MyCustomStrategy(StepStrategy):
    """Strategy for creating custom function steps (MCP-compliant)."""

    async def create_step(self, config: Dict[str, Any]) -> Callable:
        """Create a custom step from configuration.

        Args:
            config: Configuration dictionary containing 'step_function' key.

        Returns:
            The callable step function.

        Raises:
            KeyError: If 'step_function' key is missing from config.
            ValueError: If 'step_function' is not callable.
        """
        if "step_function" not in config:
            raise KeyError("step_function")
        step_func = config["step_function"]
        if not callable(step_func):
            raise ValueError("step_function must be callable")
        return step_func

    def get_node_name(self, index: int) -> str:
        """Generate node name for custom step."""
        return f"custom_step_{index}"
