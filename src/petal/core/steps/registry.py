"""Registry for step creation strategies."""

from threading import Lock
from typing import Callable, Dict, Type

from petal.core.config.agent import StepConfig
from petal.core.steps.base import StepStrategy
from petal.core.steps.custom import CustomStepStrategy
from petal.core.steps.llm import LLMStepStrategy


class StepRegistry:
    """Registry for step creation strategies."""

    def __init__(self):
        self._strategies: Dict[str, Type[StepStrategy]] = {}
        self._lock = Lock()
        self._register_defaults()

    def register(self, name: str, strategy: Type[StepStrategy]) -> None:
        """Register a step strategy.

        Args:
            name: The name of the step type.
            strategy: The StepStrategy class to register.
        """
        with self._lock:
            self._strategies[name] = strategy

    def get_strategy(self, name: str) -> StepStrategy:
        """Get a step strategy by name.

        Args:
            name: The name of the step type.

        Returns:
            An instance of the registered StepStrategy.

        Raises:
            ValueError: If the step type is not registered.
        """
        with self._lock:
            if name not in self._strategies:
                raise ValueError(f"Unknown step type: {name}")
            return self._strategies[name]()

    async def create_step(self, step_config: StepConfig) -> Callable:
        """Create a step callable from a step configuration.

        Args:
            step_config: The step configuration containing strategy type and config.

        Returns:
            A callable step function.

        Raises:
            ValueError: If the step type is not registered.
        """
        strategy = self.get_strategy(step_config.strategy_type)
        return await strategy.create_step(step_config.config)

    def validate_strategy(self, name: str) -> None:
        """Validate that a step strategy exists in the registry.

        Args:
            name: The name of the step type to validate.

        Raises:
            ValueError: If the step type is not registered.
        """
        with self._lock:
            if name not in self._strategies:
                raise ValueError(f"Unknown step type: {name}")

    def _register_defaults(self) -> None:
        """Register built-in strategies."""
        from petal.core.steps.react import ReactStepStrategy
        from petal.core.steps.tool import ToolStepStrategy

        self.register("llm", LLMStepStrategy)
        self.register("custom", CustomStepStrategy)
        self.register("tool", ToolStepStrategy)
        self.register("react", ReactStepStrategy)
