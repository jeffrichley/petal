"""Base classes for step creation strategies."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict


class StepStrategy(ABC):
    """Abstract base class for step creation strategies."""

    @abstractmethod
    def create_step(self, config: Dict[str, Any]) -> Callable:
        """Create a step callable from configuration.

        Args:
            config: Configuration dictionary containing step parameters.

        Returns:
            A callable step function that can be executed.
        """
        pass

    @abstractmethod
    def get_node_name(self, index: int) -> str:
        """Generate a node name for the step at the given index.

        Args:
            index: The index of the step in the sequence.

        Returns:
            A string representing the node name for this step.
        """
        pass


class MyCustomStrategy(StepStrategy):
    """Concrete implementation of StepStrategy for testing and example purposes."""

    def create_step(self, config: Dict[str, Any]) -> Callable:
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

        step_function = config["step_function"]
        if not callable(step_function):
            raise ValueError("step_function must be callable")

        return step_function

    def get_node_name(self, index: int) -> str:
        """Generate node name for custom step.

        Args:
            index: The index of the step in the sequence.

        Returns:
            A string in the format "custom_step_{index}".
        """
        return f"custom_step_{index}"
