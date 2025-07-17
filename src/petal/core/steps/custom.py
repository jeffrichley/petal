"""Custom step strategy for arbitrary callable functions."""

from typing import Any, Callable, Dict

from petal.core.steps.base import StepStrategy


class CustomStepStrategy(StepStrategy):
    """Strategy for creating custom function steps."""

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
        step_func = config.get("step_function")
        if not callable(step_func):
            raise ValueError("Custom step must be callable")
        return step_func

    def get_node_name(self, index: int) -> str:
        """Generate node name for custom step."""
        return f"custom_step_{index}"
