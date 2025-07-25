"""Tests for the plugin system."""

import os
import tempfile
from typing import Any, Dict, Type

import pytest

from petal.core.plugins.base import PluginManager, StepPlugin
from petal.core.steps.base import StepStrategy


@pytest.fixture
def real_strategy():
    class _RealStrategy(StepStrategy):
        def create_step(self, _config: Dict[str, Any]):
            return lambda x: f"processed_{x}"

        def get_node_name(self, index: int) -> str:
            return f"real_step_{index}"

    return _RealStrategy


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

    def test_register_plugin(self, real_strategy):
        """Test registering a plugin."""
        manager = PluginManager()

        class MockPlugin(StepPlugin):
            def get_name(self) -> str:
                return "test"

            def get_strategy(self) -> Type[StepStrategy]:
                return real_strategy

            def get_config_schema(self) -> Dict[str, Any]:
                return {}

        plugin = MockPlugin()
        manager.register(plugin)

        assert "test" in manager._plugins
        assert manager._plugins["test"] == plugin

    def test_get_plugin_success(self, real_strategy):
        """Test successfully retrieving a registered plugin."""
        manager = PluginManager()

        class MockPlugin(StepPlugin):
            def get_name(self) -> str:
                return "test"

            def get_strategy(self) -> Type[StepStrategy]:
                return real_strategy

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

    def test_register_duplicate_plugin(self):  # noqa: C901
        """Test that registering duplicate plugin overwrites the old one."""
        manager = PluginManager()

        class RealStrategy1(StepStrategy):
            """First strategy implementation."""

            def create_step(self, _config: Dict[str, Any]):
                return lambda x: f"version1_{x}"

            def get_node_name(self, index: int) -> str:
                return f"step_v1_{index}"

        class RealStrategy2(StepStrategy):
            """Second strategy implementation."""

            def create_step(self, _config: Dict[str, Any]):
                return lambda x: f"version2_{x}"

            def get_node_name(self, index: int) -> str:
                return f"step_v2_{index}"

        class MockPlugin1(StepPlugin):
            def get_name(self) -> str:
                return "test"

            def get_strategy(self) -> Type[StepStrategy]:
                return RealStrategy1

            def get_config_schema(self) -> Dict[str, Any]:
                return {"version": 1}

        class MockPlugin2(StepPlugin):
            def get_name(self) -> str:
                return "test"

            def get_strategy(self) -> Type[StepStrategy]:
                return RealStrategy2

            def get_config_schema(self) -> Dict[str, Any]:
                return {"version": 2}

        plugin1 = MockPlugin1()
        plugin2 = MockPlugin2()

        manager.register(plugin1)
        assert manager.get_plugin("test").get_config_schema()["version"] == 1

        manager.register(plugin2)
        assert manager.get_plugin("test").get_config_schema()["version"] == 2

    @pytest.mark.asyncio
    async def test_discover_plugins_with_real_module(self):
        """Test plugin discovery with a real temporary module."""
        manager = PluginManager()

        # Create a temporary module with real plugins
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the module directory
            module_dir = os.path.join(temp_dir, "test_plugin_package")
            os.makedirs(module_dir)

            # Create __init__.py with plugins
            init_content = '''
"""Test plugin package."""

from petal.core.plugins.base import StepPlugin
from petal.core.steps.base import StepStrategy


class TestStrategy(StepStrategy):
    """Test strategy implementation."""

    async def create_step(self, config):
        return lambda x: f"test_processed_{x}"

    def get_node_name(self, index):
        return f"test_step_{index}"


class TestPlugin1(StepPlugin):
    """First test plugin."""

    def get_name(self):
        return "test_plugin_1"

    def get_strategy(self):
        return TestStrategy

    def get_config_schema(self):
        return {"type": "test1"}


class TestPlugin2(StepPlugin):
    """Second test plugin."""

    def get_name(self):
        return "test_plugin_2"

    def get_strategy(self):
        return TestStrategy

    def get_config_schema(self):
        return {"type": "test2"}


PLUGINS = [TestPlugin1(), TestPlugin2()]
'''

            with open(os.path.join(module_dir, "__init__.py"), "w") as f:
                f.write(init_content)

            # Add the temp directory to Python path and discover plugins
            import sys

            sys.path.insert(0, temp_dir)

            try:
                manager.discover("test_plugin_package")

                # Verify plugins were discovered and registered
                assert "test_plugin_1" in manager._plugins
                assert "test_plugin_2" in manager._plugins

                # Test that the plugins work correctly
                plugin1 = manager.get_plugin("test_plugin_1")
                plugin2 = manager.get_plugin("test_plugin_2")

                assert plugin1.get_name() == "test_plugin_1"
                assert plugin2.get_name() == "test_plugin_2"

                # Test strategy integration
                strategy1 = plugin1.get_strategy()
                strategy2 = plugin2.get_strategy()

                assert strategy1 == strategy2  # Same strategy class

                # Test actual step creation
                step = await strategy1().create_step({})
                result = step("hello")
                assert result == "test_processed_hello"

                # Test node name generation
                node_name = strategy1().get_node_name(0)
                assert node_name == "test_step_0"

            finally:
                # Clean up
                sys.path.remove(temp_dir)

    def test_discover_plugins_no_plugins_attribute(self):
        """Test discovery when package has no PLUGINS attribute."""
        manager = PluginManager()

        # Create a temporary module without PLUGINS attribute
        with tempfile.TemporaryDirectory() as temp_dir:
            module_dir = os.path.join(temp_dir, "test_empty_package")
            os.makedirs(module_dir)

            # Create __init__.py without PLUGINS
            init_content = '''
"""Test package without plugins."""

def some_function():
    return "hello"
'''

            with open(os.path.join(module_dir, "__init__.py"), "w") as f:
                f.write(init_content)

            import sys

            sys.path.insert(0, temp_dir)

            try:
                # Should not raise an error
                manager.discover("test_empty_package")

                # No plugins should be registered
                assert len(manager._plugins) == 0

            finally:
                sys.path.remove(temp_dir)

    def test_discover_plugins_import_error(self):
        """Test discovery when package import fails."""
        manager = PluginManager()

        # Try to discover a non-existent package
        manager.discover("definitely_nonexistent_package_12345")

        # No plugins should be registered
        assert len(manager._plugins) == 0

    @pytest.mark.asyncio
    async def test_discover_plugins_with_mixed_content(self):
        """Test discovery when PLUGINS list contains non-StepPlugin items."""
        manager = PluginManager()

        # Create a temporary module with mixed content in PLUGINS
        with tempfile.TemporaryDirectory() as temp_dir:
            module_dir = os.path.join(temp_dir, "test_mixed_package")
            os.makedirs(module_dir)

            # Create __init__.py with mixed content
            init_content = '''
"""Test package with mixed PLUGINS content."""

from petal.core.plugins.base import StepPlugin
from petal.core.steps.base import StepStrategy


class TestStrategy(StepStrategy):
    """Test strategy implementation."""

    async def create_step(self, config):
        return lambda x: f"mixed_processed_{x}"

    def get_node_name(self, index):
        return f"mixed_step_{index}"


class ValidPlugin(StepPlugin):
    """Valid plugin implementation."""

    def get_name(self):
        return "valid_plugin"

    def get_strategy(self):
        return TestStrategy

    def get_config_schema(self):
        return {"type": "valid"}


# Mixed content - only ValidPlugin should be registered
PLUGINS = [
    ValidPlugin(),
    "not_a_plugin",  # String, not a StepPlugin
    42,              # Integer, not a StepPlugin
    None,            # None, not a StepPlugin
    {"key": "value"}, # Dict, not a StepPlugin
]
'''

            with open(os.path.join(module_dir, "__init__.py"), "w") as f:
                f.write(init_content)

            import sys

            sys.path.insert(0, temp_dir)

            try:
                manager.discover("test_mixed_package")

                # Only the valid plugin should be registered
                assert len(manager._plugins) == 1
                assert "valid_plugin" in manager._plugins

                # Verify the plugin works correctly
                plugin = manager.get_plugin("valid_plugin")
                assert plugin.get_name() == "valid_plugin"

                # Test strategy integration
                strategy = plugin.get_strategy()
                step = await strategy().create_step({})
                result = step("hello")
                assert result == "mixed_processed_hello"

            finally:
                sys.path.remove(temp_dir)

    def test_thread_safety(self, real_strategy):
        """Test that plugin registration is thread-safe."""
        import threading

        manager = PluginManager()
        results = []

        def register_plugin(plugin_name: str):
            """Register a plugin in a separate thread."""

            class MockPlugin(StepPlugin):
                """Mock plugin for thread safety testing."""

                def get_name(self) -> str:
                    return plugin_name

                def get_strategy(self) -> Type[StepStrategy]:
                    return real_strategy

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

    def test_list_plugins(self, real_strategy):
        """Test listing all registered plugins."""
        manager = PluginManager()

        class MockPlugin1(StepPlugin):
            def get_name(self) -> str:
                return "plugin1"

            def get_strategy(self) -> Type[StepStrategy]:
                return real_strategy

            def get_config_schema(self) -> Dict[str, Any]:
                return {}

        class MockPlugin2(StepPlugin):
            def get_name(self) -> str:
                return "plugin2"

            def get_strategy(self) -> Type[StepStrategy]:
                return real_strategy

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
