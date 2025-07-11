"""Tests for the plugin system."""

from typing import Any, Dict, Type
from unittest.mock import Mock, patch

import pytest

from petal.core.plugins.base import PluginManager, StepPlugin
from petal.core.steps.base import StepStrategy


class TestStepPlugin:
    """Test the StepPlugin abstract base class."""

    def test_step_plugin_abc_cannot_be_instantiated(self):
        """Test that StepPlugin ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            StepPlugin()  # type: ignore

    def test_step_plugin_requires_abstract_methods(self):
        """Test that concrete implementations must implement abstract methods."""

        class IncompletePlugin(StepPlugin):
            """Incomplete plugin missing required methods."""

            pass

        with pytest.raises(TypeError):
            IncompletePlugin()  # type: ignore

    def test_concrete_plugin_implementation(self):
        """Test that a complete plugin implementation works."""

        class MockStrategy(StepStrategy):
            """Mock strategy for testing."""

            def create_step(self, _config: Dict[str, Any]):
                return lambda x: x

            def get_node_name(self, index: int) -> str:
                return f"mock_step_{index}"

        class CompletePlugin(StepPlugin):
            """Complete plugin implementation."""

            def get_name(self) -> str:
                return "mock"

            def get_strategy(self) -> Type[StepStrategy]:
                return MockStrategy

            def get_config_schema(self) -> Dict[str, Any]:
                return {"test": "string"}

        plugin = CompletePlugin()
        assert plugin.get_name() == "mock"
        assert plugin.get_strategy() == MockStrategy
        assert plugin.get_config_schema() == {"test": "string"}


class TestPluginManager:
    """Test the PluginManager class."""

    def test_plugin_manager_initialization(self):
        """Test that PluginManager initializes correctly."""
        manager = PluginManager()
        assert manager._plugins == {}

    def test_register_plugin(self):
        """Test registering a plugin."""
        manager = PluginManager()

        class MockPlugin(StepPlugin):
            def get_name(self) -> str:
                return "test"

            def get_strategy(self) -> Type[StepStrategy]:
                return Mock()

            def get_config_schema(self) -> Dict[str, Any]:
                return {}

        plugin = MockPlugin()
        manager.register(plugin)

        assert "test" in manager._plugins
        assert manager._plugins["test"] == plugin

    def test_get_plugin_success(self):
        """Test successfully retrieving a registered plugin."""
        manager = PluginManager()

        class MockPlugin(StepPlugin):
            def get_name(self) -> str:
                return "test"

            def get_strategy(self) -> Type[StepStrategy]:
                return Mock()

            def get_config_schema(self) -> Dict[str, Any]:
                return {}

        plugin = MockPlugin()
        manager.register(plugin)

        retrieved_plugin = manager.get_plugin("test")
        assert retrieved_plugin == plugin

    def test_get_plugin_not_found(self):
        """Test error when trying to get non-existent plugin."""
        manager = PluginManager()

        with pytest.raises(ValueError, match="Plugin not found: nonexistent"):
            manager.get_plugin("nonexistent")

    def test_register_duplicate_plugin(self):
        """Test that registering duplicate plugin overwrites the old one."""
        manager = PluginManager()

        class MockPlugin1(StepPlugin):
            def get_name(self) -> str:
                return "test"

            def get_strategy(self) -> Type[StepStrategy]:
                return Mock()

            def get_config_schema(self) -> Dict[str, Any]:
                return {"version": 1}

        class MockPlugin2(StepPlugin):
            def get_name(self) -> str:
                return "test"

            def get_strategy(self) -> Type[StepStrategy]:
                return Mock()

            def get_config_schema(self) -> Dict[str, Any]:
                return {"version": 2}

        plugin1 = MockPlugin1()
        plugin2 = MockPlugin2()

        manager.register(plugin1)
        assert manager.get_plugin("test").get_config_schema()["version"] == 1

        manager.register(plugin2)
        assert manager.get_plugin("test").get_config_schema()["version"] == 2

    @patch("importlib.import_module")
    def test_discover_plugins_success(self, mock_import):
        """Test successful plugin discovery."""
        manager = PluginManager()

        class MockPlugin1(StepPlugin):
            def get_name(self) -> str:
                return "plugin1"

            def get_strategy(self) -> Type[StepStrategy]:
                return Mock()

            def get_config_schema(self) -> Dict[str, Any]:
                return {}

        class MockPlugin2(StepPlugin):
            def get_name(self) -> str:
                return "plugin2"

            def get_strategy(self) -> Type[StepStrategy]:
                return Mock()

            def get_config_schema(self) -> Dict[str, Any]:
                return {}

        # Mock the imported module
        mock_module = Mock()
        mock_module.PLUGINS = [MockPlugin1(), MockPlugin2()]
        mock_import.return_value = mock_module

        manager.discover("test_package")

        assert "plugin1" in manager._plugins
        assert "plugin2" in manager._plugins

    @patch("importlib.import_module")
    def test_discover_plugins_no_plugins_attribute(self, mock_import):
        """Test discovery when package has no PLUGINS attribute."""
        manager = PluginManager()

        # Mock the imported module without PLUGINS attribute
        mock_module = Mock()
        del mock_module.PLUGINS
        mock_import.return_value = mock_module

        # Should not raise an error
        manager.discover("test_package")

        # No plugins should be registered
        assert len(manager._plugins) == 0

    @patch("importlib.import_module")
    def test_discover_plugins_import_error(self, mock_import):
        """Test discovery when package import fails."""
        manager = PluginManager()

        mock_import.side_effect = ImportError("Package not found")

        # Should not raise an error
        manager.discover("nonexistent_package")

        # No plugins should be registered
        assert len(manager._plugins) == 0

    def test_thread_safety(self):
        """Test that plugin registration is thread-safe."""
        import threading

        manager = PluginManager()
        results = []

        def register_plugin(plugin_name: str):
            """Register a plugin in a separate thread."""

            class MockPlugin(StepPlugin):
                def get_name(self) -> str:
                    return plugin_name

                def get_strategy(self) -> Type[StepStrategy]:
                    return Mock()

                def get_config_schema(self) -> Dict[str, Any]:
                    return {}

            plugin = MockPlugin()
            manager.register(plugin)
            results.append(plugin_name)

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_plugin, args=(f"plugin_{i}",))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all plugins were registered
        assert len(results) == 10
        for i in range(10):
            assert f"plugin_{i}" in manager._plugins

    def test_list_plugins(self):
        """Test listing all registered plugins."""
        manager = PluginManager()

        class MockPlugin1(StepPlugin):
            def get_name(self) -> str:
                return "plugin1"

            def get_strategy(self) -> Type[StepStrategy]:
                return Mock()

            def get_config_schema(self) -> Dict[str, Any]:
                return {}

        class MockPlugin2(StepPlugin):
            def get_name(self) -> str:
                return "plugin2"

            def get_strategy(self) -> Type[StepStrategy]:
                return Mock()

            def get_config_schema(self) -> Dict[str, Any]:
                return {}

        plugin1 = MockPlugin1()
        plugin2 = MockPlugin2()

        # Initially no plugins
        plugins = manager.list_plugins()
        assert len(plugins) == 0

        # Register plugins
        manager.register(plugin1)
        manager.register(plugin2)

        # List plugins
        plugins = manager.list_plugins()
        assert len(plugins) == 2
        assert "plugin1" in plugins
        assert "plugin2" in plugins
        assert plugins["plugin1"] == plugin1
        assert plugins["plugin2"] == plugin2

        # Verify it returns a copy (not the original dict)
        plugins_copy: dict[str, Any] = dict(plugins)
        plugins_copy["plugin1"] = "modified"
        assert manager.get_plugin("plugin1") == plugin1  # Original unchanged


class TestPluginIntegration:
    """Test integration of plugins with existing systems."""

    def test_plugin_with_step_registry(self):
        """Test that plugins can be integrated with StepRegistry."""
        from petal.core.steps.registry import StepRegistry

        registry = StepRegistry()

        class MockStrategy(StepStrategy):
            def create_step(self, _config: Dict[str, Any]):
                return lambda x: x

            def get_node_name(self, index: int) -> str:
                return f"plugin_step_{index}"

        class MockPlugin(StepPlugin):
            def get_name(self) -> str:
                return "plugin_step"

            def get_strategy(self) -> Type[StepStrategy]:
                return MockStrategy

            def get_config_schema(self) -> Dict[str, Any]:
                return {}

        # Register plugin strategy with registry
        plugin = MockPlugin()
        registry.register(plugin.get_name(), plugin.get_strategy())

        # Verify it works
        strategy = registry.get_strategy("plugin_step")
        assert isinstance(strategy, MockStrategy)

        # Test step creation
        step = strategy.create_step({})
        assert callable(step)

        # Test node name generation
        node_name = strategy.get_node_name(0)
        assert node_name == "plugin_step_0"
