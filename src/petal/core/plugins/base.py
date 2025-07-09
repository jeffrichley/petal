"""Base classes for the plugin system."""

import importlib
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Dict, Type

from petal.core.steps.base import StepStrategy


class StepPlugin(ABC):
    """Abstract base class for step type plugins."""

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this step type.

        Returns:
            The name of the step type that this plugin provides.
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_strategy(self) -> Type[StepStrategy]:
        """Get the strategy class for this step type.

        Returns:
            The StepStrategy class that implements this step type.
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """Get the configuration schema for this step type.

        Returns:
            A dictionary describing the configuration schema for this step type.
        """
        pass  # pragma: no cover


class PluginManager:
    """Manager for step type plugins."""

    def __init__(self):
        """Initialize the plugin manager."""
        self._plugins: Dict[str, StepPlugin] = {}
        self._lock = Lock()

    def register(self, plugin: StepPlugin) -> None:
        """Register a plugin.

        Args:
            plugin: The plugin to register.
        """
        with self._lock:
            self._plugins[plugin.get_name()] = plugin

    def discover(self, package_name: str) -> None:
        """Discover plugins in a package.

        Args:
            package_name: The name of the package to search for plugins.
        """
        try:
            module = importlib.import_module(package_name)
            if hasattr(module, "PLUGINS"):
                for plugin in module.PLUGINS:
                    if isinstance(plugin, StepPlugin):
                        self.register(plugin)
        except ImportError:
            # Silently ignore import errors during discovery
            pass

    def get_plugin(self, name: str) -> StepPlugin:
        """Get a plugin by name.

        Args:
            name: The name of the plugin to retrieve.

        Returns:
            The plugin with the given name.

        Raises:
            ValueError: If the plugin is not found.
        """
        with self._lock:
            if name not in self._plugins:
                raise ValueError(f"Plugin not found: {name}")
            return self._plugins[name]

    def list_plugins(self) -> Dict[str, StepPlugin]:
        """List all registered plugins.

        Returns:
            A dictionary mapping plugin names to plugin instances.
        """
        with self._lock:
            return self._plugins.copy()
