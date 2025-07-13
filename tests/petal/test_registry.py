import asyncio
import threading
from typing import Optional

import pytest
from langchain_core.tools import BaseTool
from petal.core.registry import DiscoveryStrategy, ToolRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the ToolRegistry singleton before each test."""
    registry = ToolRegistry()
    registry._reset_for_testing()
    yield
    registry._reset_for_testing()


def test_tool_registry_singleton():
    """Test that ToolRegistry maintains singleton pattern."""
    registry1 = ToolRegistry()
    registry2 = ToolRegistry()
    assert registry1 is registry2

    # Test thread safety
    results = []

    def create_registry():
        results.append(ToolRegistry())

    threads = [threading.Thread(target=create_registry) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All should be the same instance
    assert all(r is registry1 for r in results)


@pytest.mark.asyncio
async def test_tool_registry_basic_operations():
    """Test basic add, resolve, and list operations."""
    registry = ToolRegistry()
    registry._reset_for_testing()  # Reset for clean state

    # Test empty registry
    assert registry.list() == []

    # Test adding tools
    from petal.core.decorators import petaltool

    @petaltool
    def test_tool():
        """Test tool."""
        return "test"

    registry.add("test_tool", test_tool)  # type: ignore[arg-type]

    # Test listing tools
    assert "test_tool" in registry.list()

    # Test resolving tools
    resolved_tool = await registry.resolve("test_tool")
    assert resolved_tool is not None

    # Test tool invocation using .invoke()
    result = resolved_tool.invoke({})
    assert result == "test"

    # Test non-existent tool
    with pytest.raises(KeyError):
        await registry.resolve("non_existent_tool")


@pytest.mark.asyncio
async def test_discovery_cache_mechanism():
    """Test that failed discoveries are cached to avoid repeated attempts."""
    registry = ToolRegistry()

    # First attempt should trigger discovery chain (empty for now)
    with pytest.raises(KeyError, match="Tool 'missing_tool' not found"):
        await registry.resolve("missing_tool")

    # Second attempt should fail immediately without discovery
    with pytest.raises(KeyError, match="Tool 'missing_tool' not found"):
        await registry.resolve("missing_tool")

    # Verify it's in discovery cache
    assert "missing_tool" in registry._discovery_cache
    assert registry._discovery_cache["missing_tool"] is False


@pytest.mark.asyncio
async def test_discovery_strategy_chain():
    """Test that discovery strategies are called in order."""
    registry = ToolRegistry()

    # Create mock strategies
    class MockStrategy(DiscoveryStrategy):
        def __init__(self, name: str, should_find: bool = False):
            self.name = name
            self.should_find = should_find
            self.called = False

        async def discover(self, _name: str) -> Optional[BaseTool]:
            self.called = True
            if self.should_find:
                from petal.core.decorators import petaltool

                @petaltool(_name)
                def found_tool():
                    """Found tool."""
                    return f"found by {self.name}"

                return found_tool  # type: ignore[return-value]
            return None

    strategy1 = MockStrategy("strategy1", should_find=False)
    strategy2 = MockStrategy("strategy2", should_find=True)
    strategy3 = MockStrategy("strategy3", should_find=True)

    registry.add_discovery_strategy(strategy1)
    registry.add_discovery_strategy(strategy2)
    registry.add_discovery_strategy(strategy3)

    # Should find tool in strategy2, not call strategy3
    tool = await registry.resolve("test_tool")
    result = tool.invoke({})
    assert result == "found by strategy2"
    assert strategy1.called
    assert strategy2.called
    assert not strategy3.called


@pytest.mark.asyncio
async def test_discovery_strategy_chain_no_found():
    """Test that all strategies are called when no tool is found."""
    registry = ToolRegistry()

    class MockStrategy(DiscoveryStrategy):
        def __init__(self, name: str):
            self.name = name
            self.called = False

        async def discover(self, _name: str) -> Optional[BaseTool]:
            self.called = True
            return None  # type: ignore[return-value]

    strategy1 = MockStrategy("strategy1")
    strategy2 = MockStrategy("strategy2")

    registry.add_discovery_strategy(strategy1)
    registry.add_discovery_strategy(strategy2)

    # Should call all strategies and fail
    with pytest.raises(KeyError, match="Tool 'missing_tool' not found"):
        await registry.resolve("missing_tool")

    assert strategy1.called
    assert strategy2.called


@pytest.mark.asyncio
async def test_discovery_strategy_exception_handling():
    """Test that exceptions in discovery strategies are handled gracefully."""
    registry = ToolRegistry()

    class FailingStrategy(DiscoveryStrategy):
        async def discover(self, _name: str) -> Optional[BaseTool]:
            raise RuntimeError("Discovery failed")

    registry.add_discovery_strategy(FailingStrategy())

    # Should continue to next strategy (none in this case) and fail
    with pytest.raises(KeyError, match="Tool 'test_tool' not found"):
        await registry.resolve("test_tool")


@pytest.mark.asyncio
async def test_tool_registry_fluent_interface():
    """Test that add() returns self for fluent interface."""
    registry = ToolRegistry()

    from petal.core.decorators import petaltool

    @petaltool
    def test_tool():
        """Test tool."""
        return "test"

    result = registry.add("test_tool", test_tool)
    assert result is registry

    # Test chaining
    @petaltool
    def another_tool():
        """Another test tool."""
        return "another"

    registry.add("another_tool", another_tool)  # type: ignore[arg-type]
    registry.add("third_tool", test_tool)  # type: ignore[arg-type]
    assert len(registry.list()) == 3


@pytest.mark.asyncio
async def test_tool_registry_overwrite_tool():
    """Test that adding a tool with same name overwrites the previous one."""
    registry = ToolRegistry()
    registry._reset_for_testing()  # Reset for clean state

    from langchain.tools import tool

    @tool
    def tool1():
        """First tool."""
        return "first"

    @tool
    def tool2():
        """Second tool."""
        return "second"

    registry.add("test_tool", tool1)  # type: ignore[arg-type]
    assert len(registry.list()) == 1

    # Overwrite with second tool
    registry.add("test_tool", tool2)  # type: ignore[arg-type]
    assert len(registry.list()) == 1

    # Verify the second tool is now registered
    resolved_tool = await registry.resolve("test_tool")
    result = resolved_tool.invoke({})
    assert result == "second"


@pytest.mark.asyncio
async def test_tool_registry_async_tool():
    """Test that async tools work correctly."""
    registry = ToolRegistry()
    registry._reset_for_testing()  # Reset for clean state

    from petal.core.decorators import petaltool

    @petaltool
    async def async_tool():
        """Async test tool."""
        await asyncio.sleep(0.01)
        return "async_result"

    registry.add("async_tool", async_tool)  # type: ignore[arg-type]

    # Test resolving and invoking async tool
    resolved_tool = await registry.resolve("async_tool")
    result = await resolved_tool.ainvoke({})
    assert result == "async_result"


@pytest.mark.asyncio
async def test_tool_registry_list_sorted():
    """Test that list() returns tools in sorted order."""
    registry = ToolRegistry()
    registry._reset_for_testing()  # Reset for clean state

    from petal.core.decorators import petaltool

    @petaltool
    def zebra():
        """Zebra tool."""
        return "zebra"

    @petaltool
    def alpha():
        """Alpha tool."""
        return "alpha"

    @petaltool
    def beta():
        """Beta tool."""
        return "beta"

    registry.add("zebra", zebra)  # type: ignore[arg-type]
    registry.add("alpha", alpha)  # type: ignore[arg-type]
    registry.add("beta", beta)  # type: ignore[arg-type]

    tool_list = registry.list()
    assert tool_list == ["alpha", "beta", "zebra"]


@pytest.mark.asyncio
async def test_tool_registry_concurrent_access():
    """Test that registry works correctly under concurrent access."""
    registry = ToolRegistry()
    registry._reset_for_testing()  # Reset for clean state

    from langchain.tools import tool

    @tool
    def tool_function():
        """Test tool function."""
        return "test_result"

    def add_tool(name: str):
        # This should work with decorated tools
        registry.add(name, tool_function)  # type: ignore[arg-type]

    def resolve_tool(_name: str):
        # This should work with decorated tools
        return asyncio.run(registry.resolve(_name))

    # Test concurrent additions
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(5):
            futures.append(executor.submit(add_tool, f"tool_{i}"))

        # Wait for all additions to complete
        concurrent.futures.wait(futures)

    # Verify all tools were added
    assert len(registry.list()) == 5
    for i in range(5):
        assert f"tool_{i}" in registry.list()

    # Test concurrent resolutions
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(5):
            futures.append(executor.submit(resolve_tool, f"tool_{i}"))

        # Wait for all resolutions to complete
        results = concurrent.futures.wait(futures)

    # Verify all tools were resolved
    assert len(results.done) == 5


@pytest.mark.asyncio
async def test_discovery_cache_after_successful_discovery():
    """Test that successful discoveries are cached and don't trigger rediscovery."""
    registry = ToolRegistry()

    class SuccessfulStrategy(DiscoveryStrategy):
        def __init__(self):
            self.discover_count = 0

        async def discover(self, _name: str) -> Optional[BaseTool]:
            self.discover_count += 1
            from petal.core.decorators import petaltool

            @petaltool(_name)
            def discovered_tool():
                """Discovered tool."""
                return "discovered"

            return discovered_tool  # type: ignore[return-value]

    strategy = SuccessfulStrategy()
    registry.add_discovery_strategy(strategy)

    # First resolution should trigger discovery
    tool1 = await registry.resolve("test_tool")
    assert tool1.invoke({}) == "discovered"
    assert strategy.discover_count == 1

    # Second resolution should use cached result
    tool2 = await registry.resolve("test_tool")
    assert tool2.invoke({}) == "discovered"
    assert strategy.discover_count == 1  # Should not have called discover again


@pytest.mark.asyncio
async def test_langchain_tool_no_wrapping():
    """Test that LangChain tools are returned directly without wrapping."""
    registry = ToolRegistry()
    registry._reset_for_testing()  # Reset for clean state

    from langchain_core.tools import BaseTool

    class MockLangChainTool(BaseTool):
        name: str = "test_tool"
        description: str = "Tool test_tool"

        def _run(self, *_args, **_kwargs):
            return "invoked test_tool"

        async def _arun(self, *_args, **_kwargs):
            return "ainvoked test_tool"

    mock_tool = MockLangChainTool()
    registry.add("test_tool", mock_tool)

    # Test resolving tool
    resolved_tool = await registry.resolve("test_tool")
    assert resolved_tool is mock_tool  # Should be the same instance

    # Test sync invocation
    sync_result = resolved_tool.invoke("test")
    assert "invoked test_tool" in sync_result

    # Test async invocation
    async_result = await resolved_tool.ainvoke("test")
    assert "ainvoked test_tool" in async_result

    # Test that regular functions cannot be registered
    def regular_function():
        return "regular"

    with pytest.raises(TypeError, match="must be decorated with @tool or @petaltool"):
        registry.add("regular_tool", regular_function)  # type: ignore[arg-type]

    # Test @petaltool decorator
    from petal.core.decorators import petaltool

    @petaltool
    def search_api(query: str) -> str:
        """Search API."""
        return f"Search: {query}"

    # Should be a BaseTool
    assert isinstance(search_api, BaseTool)  # type: ignore[arg-type]
    assert search_api.name == "search_api"  # type: ignore[union-attr]
    assert "Search API" in search_api.description  # type: ignore[union-attr]

    # Test metadata
    assert hasattr(search_api, "_petal_registered")
    assert search_api._petal_registered is True
    assert hasattr(search_api, "_original_func")
    assert search_api._original_func.__name__ == "search_api"

    # Test with custom name
    @petaltool("custom_name")
    def custom_tool(query: str) -> str:
        """Custom tool."""
        return f"Custom: {query}"

    assert isinstance(custom_tool, BaseTool)  # type: ignore[arg-type]
    assert custom_tool.name == "custom_name"  # type: ignore[union-attr]

    # Test metadata preservation
    assert hasattr(custom_tool, "_petal_registered")
    assert custom_tool._petal_registered is True
    assert hasattr(custom_tool, "_original_func")
    assert custom_tool._original_func.__name__ == "custom_tool"


@pytest.mark.asyncio
async def test_petaltool_decorator_basic():
    """Test that @petaltool decorator creates LangChain tools and registers them."""
    from petal.core.decorators import petaltool

    @petaltool
    def search_api(query: str) -> str:
        """Search the API for the given query."""
        return f"Results for: {query}"

    # Should be a LangChain BaseTool
    from langchain_core.tools import BaseTool

    assert isinstance(search_api, BaseTool)  # type: ignore[arg-type]

    # Should be registered with ToolRegistry
    registry = ToolRegistry()
    assert "search_api" in registry.list()

    # Should work with LangChain invoke
    result = search_api.invoke({"query": "test"})
    assert result == "Results for: test"


@pytest.mark.asyncio
async def test_petaltool_decorator_with_name():
    """Test that @petaltool decorator works with custom name."""
    from petal.core.decorators import petaltool

    @petaltool("custom_search", return_direct=True)
    def search_api(query: str) -> str:
        """Search the API for the given query."""
        return f"Results for: {query}"

    # Should be a LangChain BaseTool
    from langchain_core.tools import BaseTool

    assert isinstance(search_api, BaseTool)  # type: ignore[arg-type]

    # Should be registered with custom name
    registry = ToolRegistry()
    assert "custom_search" in registry.list()
    assert "search_api" not in registry.list()

    # Should work with LangChain invoke
    result = search_api.invoke({"query": "test"})
    assert result == "Results for: test"


@pytest.mark.asyncio
async def test_petaltool_decorator_metadata():
    """Test that @petaltool decorator preserves metadata."""
    from petal.core.decorators import petaltool

    @petaltool
    def search_api(query: str) -> str:
        """Search the API for the given query."""
        return f"Results for: {query}"

    # Should have proper name and description
    assert search_api.name == "search_api"  # type: ignore[union-attr]
    assert "Search the API for the given query" in search_api.description  # type: ignore[union-attr]

    # Should have proper schema
    assert hasattr(search_api, "args_schema")
    schema = search_api.args_schema.model_json_schema()  # type: ignore[union-attr]
    assert "query" in schema["properties"]
    assert schema["properties"]["query"]["type"] == "string"


@pytest.mark.asyncio
async def test_tool_registry_async_tool_invocation():
    """Test that async tools are properly invoked."""
    registry = ToolRegistry()
    registry._reset_for_testing()  # Reset for clean state

    from langchain.tools import tool

    @tool
    async def async_tool_with_args(arg1: str, arg2: int = 10) -> str:
        """Async tool with arguments."""
        await asyncio.sleep(0.01)
        return f"async_result: {arg1}, {arg2}"

    registry.add("async_tool", async_tool_with_args)  # type: ignore[arg-type]

    # Test resolving and invoking async tool
    resolved_tool = await registry.resolve("async_tool")

    # Test with positional arguments
    result = await resolved_tool.ainvoke({"arg1": "test", "arg2": 5})
    assert result == "async_result: test, 5"

    # Test with default argument
    result = await resolved_tool.ainvoke({"arg1": "default_test"})
    assert result == "async_result: default_test, 10"

    # Test that the tool has proper methods
    assert hasattr(resolved_tool, "ainvoke")
    assert hasattr(resolved_tool, "invoke")


@pytest.mark.asyncio
async def test_tool_registry_sync_tool_invocation():
    """Test that sync tools are properly invoked."""
    registry = ToolRegistry()
    registry._reset_for_testing()  # Reset for clean state

    from langchain.tools import tool

    @tool
    def sync_tool_with_args(arg1: str, arg2: int = 10) -> str:
        """Sync tool with arguments."""
        return f"sync_result: {arg1}, {arg2}"

    registry.add("sync_tool", sync_tool_with_args)  # type: ignore[arg-type]

    # Test resolving and invoking sync tool
    resolved_tool = await registry.resolve("sync_tool")

    # Test with positional arguments
    result = await resolved_tool.ainvoke({"arg1": "test", "arg2": 5})
    assert result == "sync_result: test, 5"

    # Test with default argument
    result = await resolved_tool.ainvoke({"arg1": "default_test"})
    assert result == "sync_result: default_test, 10"

    # Test that the tool has proper methods
    assert hasattr(resolved_tool, "ainvoke")
    assert hasattr(resolved_tool, "invoke")
