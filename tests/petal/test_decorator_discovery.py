"""Tests for decorator-based tool discovery."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from petal.core.discovery.decorator import DecoratorDiscovery
from petal.core.discovery.module_cache import ModuleCache
from petal.core.registry import ToolRegistry


class TestModuleCache:
    """Test the ModuleCache class."""

    @pytest.mark.asyncio
    async def test_module_cache_initialization(self):
        """Test that ModuleCache initializes correctly."""
        cache = ModuleCache()
        assert cache._scanned_modules == set()
        assert cache._module_tools == {}

    @pytest.mark.asyncio
    async def test_scan_module_returns_empty_for_nonexistent_module(self):
        """Test that scanning a nonexistent module returns empty dict."""
        cache = ModuleCache()
        tools = await cache.scan_module("nonexistent_module")
        assert tools == {}

    @pytest.mark.asyncio
    async def test_scan_module_caches_results(self):
        """Test that module scanning results are cached."""
        cache = ModuleCache()

        # Mock a module with a decorated function
        mock_module = MagicMock()
        mock_function = MagicMock()
        mock_function._petaltool_metadata = {"name": "test_tool"}
        mock_module.test_function = mock_function

        with patch.dict("sys.modules", {"test_module": mock_module}):
            # First scan
            tools1 = await cache.scan_module("test_module")
            # Second scan (should use cache)
            tools2 = await cache.scan_module("test_module")

            assert tools1 == tools2
            assert "test_module" in cache._scanned_modules

    @pytest.mark.asyncio
    async def test_scan_all_modules_handles_import_errors(self):
        """Test that scan_all_modules gracefully handles import errors."""
        cache = ModuleCache()

        # Mock sys.modules with a problematic module
        with patch.dict("sys.modules", {"bad_module": None}):
            tools = await cache.scan_all_modules()
            # Should not raise an exception
            assert isinstance(tools, dict)

    @pytest.mark.asyncio
    async def test_scan_all_modules_handles_module_scanning_errors(self):
        """Test that scan_all_modules handles errors during individual module scanning."""
        cache = ModuleCache()

        # Mock a module that raises an exception when scan_module is called
        with patch.object(
            cache, "scan_module", side_effect=Exception("Module scan error")
        ):
            tools = await cache.scan_all_modules()
            # Should not raise an exception and should return empty dict
            assert isinstance(tools, dict)
            assert tools == {}

    @pytest.mark.asyncio
    async def test_scan_module_internal_handles_module_access_errors(self):
        """Test that _scan_module_internal handles errors when accessing module attributes."""
        cache = ModuleCache()

        # Create a module that will cause vars() to fail
        class ProblematicModule:
            def __getattribute__(self, name):
                if name == "__dict__":
                    raise AttributeError("Cannot access __dict__")
                return super().__getattribute__(name)

        problematic_module = ProblematicModule()

        with patch.dict("sys.modules", {"error_module": problematic_module}):
            tools = await cache._scan_module_internal("error_module")
            # Should return empty dict instead of raising exception
            assert isinstance(tools, dict)
            assert tools == {}

    @pytest.mark.asyncio
    async def test_scan_module_internal_handles_nonexistent_module(self):
        """Test that _scan_module_internal handles nonexistent modules gracefully."""
        cache = ModuleCache()

        tools = await cache._scan_module_internal("nonexistent_module")
        # Should return empty dict for nonexistent modules
        assert isinstance(tools, dict)
        assert tools == {}

    @pytest.mark.asyncio
    async def test_clear_cache_works(self):
        """Test that clear_cache resets the cache."""
        cache = ModuleCache()
        cache._scanned_modules.add("test_module")
        from unittest.mock import MagicMock

        from langchain_core.tools import BaseTool

        cache._module_tools["test_module"] = {"tool": MagicMock(spec=BaseTool)}

        cache.clear_cache()

        assert cache._scanned_modules == set()
        assert cache._module_tools == {}

    def test_is_decorated_tool_identifies_decorated_functions(self):
        """Test that _is_decorated_tool correctly identifies decorated functions."""
        cache = ModuleCache()

        # Mock decorated function
        decorated_func = MagicMock()
        decorated_func._petaltool_metadata = {"name": "test"}

        # Mock non-decorated function
        regular_func = MagicMock()
        delattr(regular_func, "_petaltool_metadata")

        # Mock non-callable object
        non_callable = "not a function"

        assert cache._is_decorated_tool(decorated_func) is True
        assert cache._is_decorated_tool(regular_func) is False
        assert cache._is_decorated_tool(non_callable) is False

    def test_extract_tool_name_returns_correct_name(self):
        """Test that _extract_tool_name returns the correct tool name."""
        cache = ModuleCache()

        # Mock decorated function with explicit name
        decorated_func = MagicMock()
        decorated_func._petaltool_metadata = {"name": "explicit_name"}

        # Mock decorated function without explicit name
        auto_named_func = MagicMock()
        auto_named_func._petaltool_metadata = {}

        # Mock non-decorated function - explicitly remove the attribute
        regular_func = MagicMock()
        delattr(regular_func, "_petaltool_metadata")

        assert (
            cache._extract_tool_name(decorated_func, "default_name") == "explicit_name"
        )
        assert (
            cache._extract_tool_name(auto_named_func, "default_name") == "default_name"
        )
        assert cache._extract_tool_name(regular_func, "default_name") is None


class TestDecoratorDiscovery:
    """Test the DecoratorDiscovery strategy."""

    @pytest.mark.asyncio
    async def test_decorator_discovery_initialization(self):
        """Test that DecoratorDiscovery initializes correctly."""
        discovery = DecoratorDiscovery()
        assert discovery.module_cache is not None
        assert isinstance(discovery.module_cache, ModuleCache)

    @pytest.mark.asyncio
    async def test_decorator_discovery_with_custom_cache(self):
        """Test that DecoratorDiscovery can use a custom module cache."""
        custom_cache = ModuleCache()
        discovery = DecoratorDiscovery(module_cache=custom_cache)
        assert discovery.module_cache is custom_cache

    @pytest.mark.asyncio
    async def test_discover_returns_none_for_nonexistent_tool(self):
        """Test that discover returns None for nonexistent tools."""
        discovery = DecoratorDiscovery()
        tool = await discovery.discover("nonexistent_tool")
        assert tool is None

    @pytest.mark.asyncio
    async def test_discover_finds_decorated_tool(self):
        """Test that discover finds decorated tools."""
        # Mock module cache to return a tool
        mock_cache = AsyncMock()
        mock_tool = MagicMock()
        mock_cache.scan_all_modules.return_value = {"test_tool": mock_tool}

        discovery = DecoratorDiscovery(module_cache=mock_cache)
        tool = await discovery.discover("test_tool")

        assert tool is mock_tool
        mock_cache.scan_all_modules.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_handles_module_cache_errors(self):
        """Test that discover handles module cache errors gracefully."""
        # Mock module cache to raise an exception
        mock_cache = AsyncMock()
        mock_cache.scan_all_modules.side_effect = Exception("Cache error")

        discovery = DecoratorDiscovery(module_cache=mock_cache)
        tool = await discovery.discover("test_tool")

        # Should return None, not raise exception
        assert tool is None


class TestDecoratorDiscoveryIntegration:
    """Integration tests for decorator discovery with ToolRegistry."""

    @pytest.mark.asyncio
    async def test_decorator_discovery_integration_with_registry(self):
        """Test that DecoratorDiscovery integrates with ToolRegistry."""
        registry = ToolRegistry()
        discovery = DecoratorDiscovery()

        # Add discovery strategy to registry
        registry.add_discovery_strategy(discovery)

        # Mock module cache to return a tool
        mock_cache = AsyncMock()
        mock_tool = MagicMock()
        mock_cache.scan_all_modules.return_value = {"test_tool": mock_tool}
        discovery.module_cache = mock_cache

        # Try to resolve a tool
        tool = await registry.resolve("test_tool")

        assert tool is mock_tool
        assert "test_tool" in registry.list()

    @pytest.mark.asyncio
    async def test_decorator_discovery_chain_of_responsibility(self):
        """Test that DecoratorDiscovery works in discovery chain."""
        registry = ToolRegistry()
        discovery = DecoratorDiscovery()

        # Add discovery strategy to registry
        registry.add_discovery_strategy(discovery)

        # Mock module cache to return a tool
        mock_cache = AsyncMock()
        mock_tool = MagicMock()
        mock_cache.scan_all_modules.return_value = {"test_tool": mock_tool}
        discovery.module_cache = mock_cache

        # Try to resolve a tool that's not in direct registry
        tool = await registry.resolve("test_tool")

        # Check that we got a tool (don't compare mock objects directly)
        assert tool is not None
        # Tool should now be in direct registry
        assert "test_tool" in registry.list()

    @pytest.mark.asyncio
    async def test_decorator_discovery_cache_prevents_repeated_scans(self):
        """Test that discovery cache prevents repeated module scans."""
        registry = ToolRegistry()
        discovery = DecoratorDiscovery()

        # Add discovery strategy to registry
        registry.add_discovery_strategy(discovery)

        # Mock module cache
        mock_cache = AsyncMock()
        mock_cache.scan_all_modules.return_value = {}
        discovery.module_cache = mock_cache

        # Try to resolve the same tool twice
        from contextlib import suppress

        with suppress(KeyError):
            await registry.resolve("nonexistent_tool")

        with suppress(KeyError):
            await registry.resolve("nonexistent_tool")

        # Should only scan once due to discovery cache
        assert mock_cache.scan_all_modules.call_count == 1
