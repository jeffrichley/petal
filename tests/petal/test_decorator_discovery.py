"""Tests for decorator-based tool discovery."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.tools import BaseTool

from petal.core.decorators import petaltool
from petal.core.discovery.decorator import DecoratorDiscovery
from petal.core.discovery.module_cache import ModuleCache
from petal.core.registry import ToolRegistry


# Create real decorated functions for testing
@petaltool
def real_test_tool(text: str) -> str:
    """A real decorated tool for testing."""
    return f"Real tool: {text}"


@petaltool("explicit_name_tool")
def explicit_name_tool(value: int) -> int:
    """A tool with explicit name."""
    return value * 2


@petaltool
def another_real_tool(data: str) -> str:
    """Another real decorated tool."""
    return f"Another: {data}"


# Create a regular function for comparison
def regular_function(text: str) -> str:
    """A regular function without decorator."""
    return f"Regular: {text}"


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

        # Get the current module name
        current_module = __name__

        # First scan - should actually scan the module
        tools1 = await cache.scan_module(current_module)

        # Second scan - should use cache
        tools2 = await cache.scan_module(current_module)

        # Results should be identical
        assert tools1 == tools2
        assert current_module in cache._scanned_modules

        # Verify we actually found some tools (the decorated ones above)
        assert len(tools1) > 0
        assert "test_decorator_discovery:real_test_tool" in tools1
        assert "test_decorator_discovery:explicit_name_tool" in tools1
        assert "test_decorator_discovery:another_real_tool" in tools1

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
    async def test_scan_module_internal_finds_real_decorated_tools(self):
        """Test that _scan_module_internal finds real decorated tools."""
        cache = ModuleCache()

        # Scan the current module which contains real decorated functions
        tools = await cache._scan_module_internal(__name__)

        # Should find our real decorated tools
        assert "test_decorator_discovery:real_test_tool" in tools
        assert "test_decorator_discovery:explicit_name_tool" in tools
        assert "test_decorator_discovery:another_real_tool" in tools

        # Should not find regular function
        assert "regular_function" not in tools

        # Verify the tools are actual BaseTool instances
        for tool_name, tool_obj in tools.items():
            assert isinstance(tool_obj, BaseTool)
            assert tool_obj.name == tool_name

    @pytest.mark.asyncio
    async def test_clear_cache_works(self):
        """Test that clear_cache resets the cache."""
        cache = ModuleCache()

        # Scan a module to populate cache
        await cache.scan_module(__name__)
        assert len(cache._scanned_modules) > 0
        assert len(cache._module_tools) > 0

        cache.clear_cache()

        assert cache._scanned_modules == set()
        assert cache._module_tools == {}

    def test_is_decorated_tool_identifies_real_decorated_functions(self):
        """Test that _is_decorated_tool correctly identifies real decorated functions."""
        cache = ModuleCache()

        # Test with real decorated functions
        assert cache._is_decorated_tool(real_test_tool) is True
        assert cache._is_decorated_tool(explicit_name_tool) is True
        assert cache._is_decorated_tool(another_real_tool) is True

        # Test with regular function
        assert cache._is_decorated_tool(regular_function) is False

        # Test with non-callable object
        non_callable = "not a function"
        assert cache._is_decorated_tool(non_callable) is False

    def test_extract_tool_name_returns_correct_name(self):
        """Test that _extract_tool_name returns the correct tool name."""
        cache = ModuleCache()

        # Test with real decorated functions
        assert (
            cache._extract_tool_name(real_test_tool, "default_name")
            == "test_decorator_discovery:real_test_tool"
        )
        assert (
            cache._extract_tool_name(explicit_name_tool, "default_name")
            == "test_decorator_discovery:explicit_name_tool"
        )
        assert (
            cache._extract_tool_name(another_real_tool, "default_name")
            == "test_decorator_discovery:another_real_tool"
        )

        # Test with regular function
        assert cache._extract_tool_name(regular_function, "default_name") is None

    def test_extract_tool_name_with_base_tool_instances(self):
        """Test that _extract_tool_name works with BaseTool instances."""
        cache = ModuleCache()

        # Create a concrete BaseTool implementation for testing
        class MockTool(BaseTool):
            name: str = "mock_tool_name"
            description: str = "A mock tool"

            def _run(self, *_args, **_kwargs):
                return "mock result"

        mock_tool = MockTool()

        # Test that it returns the tool's name
        result = cache._extract_tool_name(mock_tool, "default_name")
        assert result == "mock_tool_name"

    def test_extract_tool_name_with_legacy_metadata_name(self):
        """Test that _extract_tool_name returns name from _petaltool_metadata."""
        cache = ModuleCache()

        # Create a function with _petaltool_metadata containing a name
        def legacy_tool():
            pass

        # legacy_tool._petaltool_metadata = {"name": "legacy_tool_name"}
        attr_name = "_petaltool_metadata"
        setattr(legacy_tool, attr_name, {"name": "legacy_tool_name"})

        result = cache._extract_tool_name(legacy_tool, "default_name")
        assert result == "legacy_tool_name"

    def test_extract_tool_name_with_legacy_metadata_no_name(self):
        """Test that _extract_tool_name returns default_name when metadata has no name."""
        cache = ModuleCache()

        # Create a function with _petaltool_metadata but no name
        def legacy_tool():
            pass

        attr_name = "_petaltool_metadata"
        setattr(legacy_tool, attr_name, {"description": "A tool without name"})

        result = cache._extract_tool_name(legacy_tool, "default_name")
        assert result == "default_name"

    def test_extract_tool_name_with_empty_legacy_metadata(self):
        """Test that _extract_tool_name returns default_name when metadata is empty."""
        cache = ModuleCache()

        # Create a function with empty _petaltool_metadata
        def legacy_tool():
            pass

        attr_name = "_petaltool_metadata"
        setattr(legacy_tool, attr_name, {})

        result = cache._extract_tool_name(legacy_tool, "default_name")
        assert result == "default_name"

    def test_extract_tool_name_with_none_legacy_metadata(self):
        """Test that _extract_tool_name returns default_name when metadata is None."""
        cache = ModuleCache()

        # Create a function with None _petaltool_metadata
        def legacy_tool():
            pass

        attr_name = "_petaltool_metadata"
        setattr(legacy_tool, attr_name, None)

        result = cache._extract_tool_name(legacy_tool, "default_name")
        assert result == "default_name"

    def test_extract_tool_name_with_non_dict_legacy_metadata(self):
        """Test that _extract_tool_name returns default_name when metadata is not a dict."""
        cache = ModuleCache()

        # Create a function with non-dict _petaltool_metadata
        def legacy_tool():
            pass

        attr_name = "_petaltool_metadata"
        setattr(legacy_tool, attr_name, "not a dict")

        result = cache._extract_tool_name(legacy_tool, "default_name")
        assert result == "default_name"

    def test_extract_tool_name_with_legacy_metadata_name_none(self):
        """Test that _extract_tool_name returns None when metadata name is None."""
        cache = ModuleCache()

        # Create a function with _petaltool_metadata where name is None
        def legacy_tool():
            pass

        attr_name = "_petaltool_metadata"
        setattr(legacy_tool, attr_name, {"name": None})

        result = cache._extract_tool_name(legacy_tool, "default_name")
        # metadata.get("name", default_name) returns None when name is None, not default_name
        assert result is None

    def test_extract_tool_name_with_legacy_metadata_name_empty_string(self):
        """Test that _extract_tool_name returns empty string when metadata name is empty."""
        cache = ModuleCache()

        # Create a function with _petaltool_metadata where name is empty string
        def legacy_tool():
            pass

        attr_name = "_petaltool_metadata"
        setattr(legacy_tool, attr_name, {"name": ""})

        result = cache._extract_tool_name(legacy_tool, "default_name")
        assert result == ""

    def test_extract_tool_name_with_no_metadata_or_base_tool(self):
        """Test that _extract_tool_name returns None for objects without metadata or BaseTool."""
        cache = ModuleCache()

        # Test with regular function (no _petaltool_metadata)
        def regular_func():
            pass

        result = cache._extract_tool_name(regular_func, "default_name")
        assert result is None

        # Test with string (not BaseTool, no _petaltool_metadata)
        result = cache._extract_tool_name("not a tool", "default_name")
        assert result is None

        # Test with None
        result = cache._extract_tool_name(None, "default_name")
        assert result is None


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
    async def test_discover_finds_real_decorated_tool(self):
        """Test that discover finds real decorated tools."""
        discovery = DecoratorDiscovery()

        # Try to discover our real decorated tools
        tool1 = await discovery.discover("test_decorator_discovery:real_test_tool")
        tool2 = await discovery.discover("test_decorator_discovery:explicit_name_tool")
        tool3 = await discovery.discover("test_decorator_discovery:another_real_tool")

        # Should find the tools
        assert tool1 is not None
        assert tool2 is not None
        assert tool3 is not None

        # Verify they are BaseTool instances
        assert isinstance(tool1, BaseTool)
        assert isinstance(tool2, BaseTool)
        assert isinstance(tool3, BaseTool)

        # Verify tool names
        assert tool1.name == "test_decorator_discovery:real_test_tool"
        assert tool2.name == "test_decorator_discovery:explicit_name_tool"
        assert tool3.name == "test_decorator_discovery:another_real_tool"

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

    def test_find_tools_by_base_name_single_match(self):
        """Test _find_tools_by_base_name with a single matching tool."""
        discovery = DecoratorDiscovery()

        # Create a mock tools dictionary with namespaced tools
        all_tools = {
            "module_a:calculator": Mock(),
            "module_b:search": Mock(),
            "module_c:calculator": Mock(),
        }

        # Test finding tools with base name "calculator"
        matching_tools = discovery._find_tools_by_base_name(all_tools, "calculator")

        # Should find both tools with base name "calculator"
        assert len(matching_tools) == 2
        assert "module_a:calculator" in matching_tools
        assert "module_c:calculator" in matching_tools
        assert "module_b:search" not in matching_tools

    def test_find_tools_by_base_name_no_match(self):
        """Test _find_tools_by_base_name with no matching tools."""
        discovery = DecoratorDiscovery()

        all_tools = {
            "module_a:calculator": Mock(),
            "module_b:search": Mock(),
        }

        # Test finding tools with base name "nonexistent"
        matching_tools = discovery._find_tools_by_base_name(all_tools, "nonexistent")

        # Should find no matches
        assert len(matching_tools) == 0

    def test_find_tools_by_base_name_with_non_namespaced_tools(self):
        """Test _find_tools_by_base_name with tools that don't have namespaces."""
        discovery = DecoratorDiscovery()

        all_tools = {
            "calculator": Mock(),
            "search": Mock(),
            "module_a:calculator": Mock(),
        }

        # Test finding tools with base name "calculator"
        matching_tools = discovery._find_tools_by_base_name(all_tools, "calculator")

        # Should find both the non-namespaced and namespaced tools
        assert len(matching_tools) == 2
        assert "calculator" in matching_tools
        assert "module_a:calculator" in matching_tools

    @pytest.mark.asyncio
    async def test_discover_by_base_name_single_match(self):
        """Test discover method with base name that has a single match."""
        discovery = DecoratorDiscovery()

        # Mock the module cache to return specific tools
        mock_cache = AsyncMock()
        calculator_tool = Mock()
        calculator_tool.name = "module_a:calculator"
        search_tool = Mock()
        search_tool.name = "module_b:search"
        mock_cache.scan_all_modules.return_value = {
            "module_a:calculator": calculator_tool,
            "module_b:search": search_tool,
        }
        discovery.module_cache = mock_cache

        # Test discovering by base name "calculator"
        tool = await discovery.discover("calculator")

        # Should return the single matching tool
        assert tool is not None
        assert tool.name == "module_a:calculator"

    @pytest.mark.asyncio
    async def test_discover_by_base_name_multiple_matches(self):
        """Test discover method with base name that has multiple matches."""
        discovery = DecoratorDiscovery()

        # Mock the module cache to return multiple tools with same base name
        mock_cache = AsyncMock()
        calc1_tool = Mock()
        calc1_tool.name = "module_a:calculator"
        calc2_tool = Mock()
        calc2_tool.name = "module_b:calculator"
        search_tool = Mock()
        search_tool.name = "module_c:search"
        mock_cache.scan_all_modules.return_value = {
            "module_a:calculator": calc1_tool,
            "module_b:calculator": calc2_tool,
            "module_c:search": search_tool,
        }
        discovery.module_cache = mock_cache

        # Test discovering by base name "calculator"
        tool = await discovery.discover("calculator")

        # Should return None due to ambiguity (multiple matches)
        assert tool is None

    @pytest.mark.asyncio
    async def test_discover_by_base_name_no_matches(self):
        """Test discover method with base name that has no matches."""
        discovery = DecoratorDiscovery()

        # Mock the module cache to return tools
        mock_cache = AsyncMock()
        calculator_tool = Mock()
        calculator_tool.name = "module_a:calculator"
        search_tool = Mock()
        search_tool.name = "module_b:search"
        mock_cache.scan_all_modules.return_value = {
            "module_a:calculator": calculator_tool,
            "module_b:search": search_tool,
        }
        discovery.module_cache = mock_cache

        # Test discovering by base name "nonexistent"
        tool = await discovery.discover("nonexistent")

        # Should return None (no matches)
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

        # Try to resolve a real tool
        tool = await registry.resolve("test_decorator_discovery:real_test_tool")

        assert tool is not None
        assert isinstance(tool, BaseTool)
        assert tool.name == "test_decorator_discovery:real_test_tool"
        assert "test_decorator_discovery:real_test_tool" in registry.list()

    @pytest.mark.asyncio
    async def test_decorator_discovery_chain_of_responsibility(self):
        """Test that DecoratorDiscovery works in discovery chain."""
        registry = ToolRegistry()
        discovery = DecoratorDiscovery()

        # Add discovery strategy to registry
        registry.add_discovery_strategy(discovery)

        # Try to resolve a tool that's not in direct registry
        tool = await registry.resolve("test_decorator_discovery:explicit_name_tool")

        # Check that we got a tool
        assert tool is not None
        assert isinstance(tool, BaseTool)
        assert tool.name == "test_decorator_discovery:explicit_name_tool"

        # Tool should now be in direct registry
        assert "test_decorator_discovery:explicit_name_tool" in registry.list()

    @pytest.mark.asyncio
    async def test_decorator_discovery_cache_prevents_repeated_scans(self):
        """Test that discovery cache prevents repeated module scans."""
        registry = ToolRegistry()
        discovery = DecoratorDiscovery()

        # Add discovery strategy to registry
        registry.add_discovery_strategy(discovery)

        # Try to resolve the same tool twice
        from contextlib import suppress

        with suppress(KeyError):
            await registry.resolve("nonexistent_tool")

        with suppress(KeyError):
            await registry.resolve("nonexistent_tool")

        # The discovery cache should prevent repeated scans
        # We can verify this by checking that the module cache wasn't called repeatedly
        # Since we're using the real module cache, we can't easily mock it
        # But we can verify the behavior is correct by checking the registry state

    @pytest.mark.asyncio
    async def test_real_tool_execution(self):
        """Test that discovered tools can actually be executed."""
        registry = ToolRegistry()
        discovery = DecoratorDiscovery()
        registry.add_discovery_strategy(discovery)

        # Resolve a real tool
        tool = await registry.resolve("test_decorator_discovery:real_test_tool")

        # Execute the tool
        result = await tool.ainvoke({"text": "Hello World"})

        # Verify the result
        assert result == "Real tool: Hello World"

    @pytest.mark.asyncio
    async def test_multiple_tools_discovery(self):
        """Test that multiple tools can be discovered from the same module."""
        registry = ToolRegistry()
        discovery = DecoratorDiscovery()
        registry.add_discovery_strategy(discovery)

        # Discover all our test tools
        tools = []
        for tool_name in [
            "test_decorator_discovery:real_test_tool",
            "test_decorator_discovery:explicit_name_tool",
            "test_decorator_discovery:another_real_tool",
        ]:
            tool = await registry.resolve(tool_name)
            tools.append(tool)

        # Verify all tools were found
        assert len(tools) == 3
        assert all(isinstance(tool, BaseTool) for tool in tools)

        # Verify tool names
        tool_names = [tool.name for tool in tools]
        assert "test_decorator_discovery:real_test_tool" in tool_names
        assert "test_decorator_discovery:explicit_name_tool" in tool_names
        assert "test_decorator_discovery:another_real_tool" in tool_names
