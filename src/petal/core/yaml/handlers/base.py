"""Base handler interface for YAML node configuration."""

from abc import ABC, abstractmethod
from typing import Callable

from petal.core.config.yaml import BaseNodeConfig


class NodeConfigHandler(ABC):
    """Abstract base class for node configuration handlers."""

    @abstractmethod
    def create_node(self, config: BaseNodeConfig) -> Callable:
        """Create a runnable node from configuration.

        Args:
            config: The node configuration

        Returns:
            A callable node function
        """
