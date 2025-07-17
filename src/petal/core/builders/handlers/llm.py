"""LLM configuration handler for step processing."""

from typing import Any, Callable, Dict

from petal.core.builders.handlers.base import StepConfigHandler
from petal.core.steps.llm import LLMStepStrategy


class LLMConfigHandler(StepConfigHandler):
    """
    Handler for LLM step configuration.

    This handler processes LLM step configurations and creates LLMStep instances
    using the LLMStepStrategy. It validates LLM-specific configuration parameters
    and ensures proper step creation.
    """

    def can_handle(self, step_type: str) -> bool:
        """
        Check if this handler can handle the given step type.

        Args:
            step_type: The type of step to check

        Returns:
            True if the step type is "llm", False otherwise
        """
        return step_type == "llm"

    async def handle(self, config: Dict[str, Any]) -> Callable:
        """
        Handle LLM step configuration and create an LLM step.

        Args:
            config: Configuration dictionary for the LLM step

        Returns:
            A callable LLM step function

        Raises:
            ValueError: If the configuration is invalid for LLM steps
        """
        # Validate that we have at least a prompt template
        if not config.get("prompt_template"):
            raise ValueError("LLM step requires a prompt_template")

        # Create LLM step using the strategy
        strategy = LLMStepStrategy()
        return await strategy.create_step(config)
