"""Abstract base class for step configuration handlers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class StepConfigHandler(ABC):
    """
    Abstract base class for step configuration handlers using Chain of Responsibility pattern.

    This class implements the Chain of Responsibility pattern, allowing handlers
    to be chained together. Each handler can either handle a specific step type
    or delegate to the next handler in the chain.
    """

    def __init__(self, next_handler: Optional["StepConfigHandler"] = None):
        """
        Initialize the handler with an optional next handler in the chain.

        Args:
            next_handler: The next handler in the chain, or None if this is the last handler
        """
        self.next_handler = next_handler

    @abstractmethod
    def can_handle(self, step_type: str) -> bool:
        """
        Check if this handler can handle the given step type.

        Args:
            step_type: The type of step to check

        Returns:
            True if this handler can handle the step type, False otherwise
        """

    @abstractmethod
    def handle(self, config: Dict[str, Any]) -> Callable:
        """
        Handle step configuration and create a callable step.

        Args:
            config: Configuration dictionary for the step

        Returns:
            A callable step function

        Raises:
            ValueError: If the configuration is invalid for this handler
        """

    def process(self, step_type: str, config: Dict[str, Any]) -> Callable:
        """
        Process configuration through the chain of responsibility.

        This method checks if the current handler can handle the step type.
        If it can, it calls the handle method. If not, it delegates to the
        next handler in the chain. If no handler can handle the step type,
        it raises a ValueError.

        Args:
            step_type: The type of step to process
            config: Configuration dictionary for the step

        Returns:
            A callable step function

        Raises:
            ValueError: If no handler in the chain can handle the step type
        """
        if self.can_handle(step_type):
            return self.handle(config)
        elif self.next_handler:
            return self.next_handler.process(step_type, config)
        else:
            raise ValueError(f"No handler found for step type: {step_type}")
