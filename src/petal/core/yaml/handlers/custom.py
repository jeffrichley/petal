"""Custom node configuration handler."""

import importlib
from typing import Any, Callable

from petal.core.config.yaml import BaseNodeConfig, CustomNodeConfig
from petal.core.yaml.handlers.base import NodeConfigHandler


class CustomNodeHandler(NodeConfigHandler):
    """Handler for creating Custom nodes from YAML configuration."""

    def create_node(self, config: BaseNodeConfig) -> Callable:
        """Create a custom node from configuration.

        Args:
            config: The custom node configuration

        Returns:
            A callable custom node function
        """
        # Cast config to CustomNodeConfig
        custom_config = (
            config
            if isinstance(config, CustomNodeConfig)
            else CustomNodeConfig(**config.__dict__)
        )

        # Import the function
        function = self._import_function(custom_config.function_path)

        # Create a wrapper function that applies parameters
        def node_function(state: Any, **kwargs) -> Any:
            """Custom node function with applied parameters."""
            # Merge state and parameters
            call_kwargs = {**custom_config.parameters, **kwargs}
            return function(state, **call_kwargs)

        return node_function

    def _import_function(self, function_path: str) -> Callable:
        """Import function from module path.

        Args:
            function_path: Python import path to function (e.g., "module.function")

        Returns:
            The imported function

        Raises:
            ImportError: If module or function cannot be imported
            ValueError: If function_path is invalid
        """
        try:
            # Split module path and function name
            if "." not in function_path:
                raise ValueError(f"Invalid function path: {function_path}")

            module_path, function_name = function_path.rsplit(".", 1)

            # Import module
            module = importlib.import_module(module_path)

            # Get function from module
            if not hasattr(module, function_name):
                raise ImportError(
                    f"Function '{function_name}' not found in module '{module_path}'"
                )

            function = getattr(module, function_name)

            if not callable(function):
                raise ValueError(f"'{function_name}' is not callable")

            return function

        except ImportError as e:
            raise ImportError(
                f"Failed to import function '{function_path}': {e}"
            ) from e
        except Exception as e:
            raise ValueError(f"Invalid function path '{function_path}': {e}") from e
