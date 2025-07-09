"""Custom configuration handler for step processing."""

from typing import Any, Callable, Dict

from petal.core.builders.handlers.base import StepConfigHandler
from petal.core.steps.custom import CustomStepStrategy


class CustomConfigHandler(StepConfigHandler):
    """
    Handler for custom step configuration.

    This handler processes custom step configurations and creates custom step functions
    from arbitrary callables. It validates that the step function is callable and
    supports both synchronous and asynchronous functions.
    """

    def can_handle(self, step_type: str) -> bool:
        """
        Check if this handler can handle the given step type.

        Args:
            step_type: The type of step to check

        Returns:
            True if the step type is "custom", False otherwise
        """
        return step_type == "custom"

    def handle(self, config: Dict[str, Any]) -> Callable:
        """
        Handle custom step configuration and create a custom step.

        Args:
            config: Configuration dictionary for the custom step

        Returns:
            A callable custom step function

        Raises:
            ValueError: If the configuration is invalid for custom steps
        """
        # Validate that we have a step function
        step_function = config.get("step_function")
        if not callable(step_function):
            raise ValueError("Custom step must be callable")

        # Create custom step using the strategy
        strategy = CustomStepStrategy()
        return strategy.create_step(config)
