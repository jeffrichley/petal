"""Factory for creating node configuration handlers."""

from typing import Dict, Type

from petal.core.yaml.handlers.base import NodeConfigHandler
from petal.core.yaml.handlers.llm import LLMNodeHandler
from petal.core.yaml.handlers.react import ReactNodeHandler


class HandlerFactory:
    """Factory for creating node configuration handlers."""

    def __init__(self):
        self._handlers: Dict[str, Type[NodeConfigHandler]] = {
            "llm": LLMNodeHandler,
            "react": ReactNodeHandler,
        }

    def get_handler(self, node_type: str) -> NodeConfigHandler:
        """Get handler for node type.

        Args:
            node_type: The type of node to get handler for

        Returns:
            An instance of the appropriate NodeConfigHandler

        Raises:
            ValueError: If the node type is not supported
        """
        if node_type not in self._handlers:
            raise ValueError(f"Unknown node type: {node_type}")
        return self._handlers[node_type]()
