import asyncio
from typing import Any

import pytest
from petal.core.tool_factory import ToolFactory


@pytest.mark.asyncio
async def test_add_and_resolve_sync_tool() -> None:
    def echo(x: str) -> str:
        return x

    tf = ToolFactory()
    tf.add("echo", echo)
    fn = tf.resolve("echo")
    result: str = await fn("hello")
    assert result == "hello"


@pytest.mark.asyncio
async def test_add_and_resolve_async_tool() -> None:
    async def async_double(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    tf = ToolFactory()
    tf.add("double", async_double)
    fn = tf.resolve("double")
    result: int = await fn(3)
    assert result == 6


def test_list_tools() -> None:
    tf = ToolFactory()
    tf.add("a", lambda: 1)
    tf.add("b", lambda: 2)
    assert tf.list() == ["a", "b"]


def test_resolve_missing_tool_raises() -> None:
    tf = ToolFactory()
    with pytest.raises(KeyError):
        tf.resolve("not_found")


@pytest.mark.asyncio
async def test_overwrite_tool() -> None:
    tf = ToolFactory()
    tf.add("dup", lambda: 1)
    tf.add("dup", lambda: 2)
    fn = tf.resolve("dup")
    result: int = await fn()
    assert result == 2


@pytest.mark.asyncio
async def test_resolve_unloaded_mcp_tool_raises() -> None:
    """Test that resolving an MCP tool before it's loaded raises an appropriate error."""
    tf = ToolFactory()

    # Try to resolve an MCP tool that hasn't been registered yet
    with pytest.raises(
        KeyError,
        match="MCP tool 'mcp:test_server:add' not found. Server 'test_server' has not been registered.",
    ):
        tf.resolve("mcp:test_server:add")


@pytest.mark.asyncio
async def test_resolve_loading_mcp_tool_raises() -> None:
    """Test that resolving an MCP tool while it's still loading raises an appropriate error."""
    tf = ToolFactory()

    # Register an MCP server but don't wait for it to load
    async def slow_resolver() -> list[Any]:
        await asyncio.sleep(0.1)  # Simulate slow loading
        return []

    tf.add_mcp("test_server", slow_resolver)

    # Try to resolve a tool while it's still loading
    with pytest.raises(
        KeyError,
        match="MCP tool 'mcp:test_server:add' is still loading. Please wait for it to complete.",
    ):
        tf.resolve("mcp:test_server:add")


@pytest.mark.asyncio
async def test_wait_for_mcp_and_resolve_tool_not_found() -> None:
    """Test that _wait_for_mcp_and_resolve raises an error when tool is not found after loading."""
    tf = ToolFactory()

    # Register an MCP server that returns no tools
    async def empty_resolver() -> list[Any]:
        return []

    tf.add_mcp("test_server", empty_resolver)

    # Wait for MCP to load
    await tf.await_mcp_loaded("test_server")

    # Try to resolve a tool that doesn't exist
    with pytest.raises(
        KeyError,
        match="MCP tool 'mcp:test_server:nonexistent' is still loading. Please wait for it to complete.",
    ):
        tf.resolve("mcp:test_server:nonexistent")


@pytest.mark.asyncio
async def test_add_mcp_tools_with_custom_resolver(
    mcp_server_config: dict[str, dict[str, object]],
) -> None:
    """Test add_mcp with custom resolver (original approach)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    tf = ToolFactory()

    async def resolver() -> list[Any]:
        client = MultiServerMCPClient(mcp_server_config)
        tools = await client.get_tools()
        return tools

    tf.add_mcp("test_server", resolver)
    await tf.await_mcp_loaded("test_server")
    tool_names = tf.list()
    assert "mcp:test_server:add" in tool_names
    assert "mcp:test_server:multiply" in tool_names

    add_tool = tf.resolve("mcp:test_server:add")
    result = await add_tool.ainvoke({"a": 2, "b": 3})  # type: ignore
    assert int(result) == 5

    multiply_tool = tf.resolve("mcp:test_server:multiply")
    result = await multiply_tool.ainvoke({"a": 2, "b": 3})  # type: ignore
    assert int(result) == 6


@pytest.mark.asyncio
async def test_add_mcp_tools_with_default_resolver(
    mcp_server_config: dict[str, dict[str, object]],
) -> None:
    """Test add_mcp with default resolver (new approach)."""
    tf = ToolFactory()
    tf.add_mcp("test_server", mcp_config=mcp_server_config)
    await tf.await_mcp_loaded("test_server")
    tool_names = tf.list()
    assert "mcp:test_server:add" in tool_names
    assert "mcp:test_server:multiply" in tool_names

    add_tool = tf.resolve("mcp:test_server:add")
    result = await add_tool.ainvoke({"a": 2, "b": 3})  # type: ignore
    assert int(result) == 5

    multiply_tool = tf.resolve("mcp:test_server:multiply")
    result = await multiply_tool.ainvoke({"a": 2, "b": 3})  # type: ignore
    assert int(result) == 6


@pytest.mark.asyncio
async def test_add_mcp_tools_missing_config_raises() -> None:
    """Test that add_mcp raises an error when resolver is None but mcp_config is not provided."""
    tf = ToolFactory()
    with pytest.raises(
        ValueError, match="mcp_config is required when resolver is None"
    ):
        tf.add_mcp("test_server")  # No resolver, no config


def test_tool_factory_init():
    """Test ToolFactory initialization."""
    tf = ToolFactory()
    # Test that a fresh factory has no tools
    assert tf.list() == []


@pytest.mark.asyncio
async def test_tool_factory_add_mcp_fluent_interface():
    """Test that add_mcp() returns self for fluent interface."""
    tf = ToolFactory()

    async def resolver():
        return []

    result = tf.add_mcp("test_server", resolver)
    assert result is tf


def test_tool_factory_list_empty():
    """Test that list() returns empty list when no tools registered."""
    tf = ToolFactory()
    assert tf.list() == []


def test_tool_factory_list_sorted():
    """Test that list() returns tools in sorted order."""
    tf = ToolFactory()
    tf.add("zebra", lambda: "zebra")
    tf.add("alpha", lambda: "alpha")
    tf.add("beta", lambda: "beta")

    assert tf.list() == ["alpha", "beta", "zebra"]


@pytest.mark.asyncio
async def test_tool_factory_resolve_regular_tool():
    """Test that resolve() works for regular (non-MCP) tools."""
    tf = ToolFactory()
    tf.add("test", lambda x: x * 2)

    tool = tf.resolve("test")
    result: int = await tool(5)
    assert result == 10


def test_tool_factory_resolve_mcp_tool_not_registered():
    """Test that resolve() raises error for MCP tool when server not registered."""
    tf = ToolFactory()

    with pytest.raises(
        KeyError,
        match="MCP tool 'mcp:test_server:tool' not found. Server 'test_server' has not been registered.",
    ):
        tf.resolve("mcp:test_server:tool")


@pytest.mark.asyncio
async def test_tool_factory_resolve_mcp_tool_loading():
    """Test that resolve() raises error for MCP tool while loading."""
    tf = ToolFactory()

    async def slow_resolver():
        await asyncio.sleep(0.1)
        return []

    tf.add_mcp("test_server", slow_resolver)

    with pytest.raises(
        KeyError,
        match="MCP tool 'mcp:test_server:tool' is still loading. Please wait for it to complete.",
    ):
        tf.resolve("mcp:test_server:tool")


@pytest.mark.asyncio
async def test_await_mcp_loaded_not_registered():
    """Test that await_mcp_loaded() works when server not registered."""
    tf = ToolFactory()
    # Should not raise when server not registered
    await tf.await_mcp_loaded("nonexistent_server")


@pytest.mark.asyncio
async def test_await_mcp_loaded_registered():
    """Test that await_mcp_loaded() waits for registered server."""
    tf = ToolFactory()

    async def fast_resolver():
        return []

    tf.add_mcp("test_server", fast_resolver)
    await tf.await_mcp_loaded("test_server")
    # Should complete without error


@pytest.mark.asyncio
async def test_tool_factory_add_mcp_with_resolver_and_config():
    """Test that add_mcp() works with both resolver and config (resolver takes precedence)."""
    tf = ToolFactory()

    async def custom_resolver():
        return []

    # Should not raise even though both resolver and config are provided
    # (resolver takes precedence)
    tf.add_mcp("test_server", custom_resolver, mcp_config={"test": {"config": "value"}})


@pytest.mark.asyncio
async def test_tool_factory_mcp_task_cleanup():
    """Test that MCP tasks are properly tracked and tools are added."""
    tf = ToolFactory()

    # Create a simple tool with a name attribute
    class SimpleTool:
        def __init__(self, name):
            self.name = name

        def __call__(self, x):
            return x

    async def resolver():
        # Return tools with proper name attributes
        return [SimpleTool("test_tool")]

    tf.add_mcp("test_server", resolver)
    # Test that we can await the server to load
    await tf.await_mcp_loaded("test_server")
    # Verify that tools were added
    assert len(tf.list()) > 0
    assert "mcp:test_server:test_tool" in tf.list()


@pytest.mark.asyncio
async def test_tool_factory_mcp_event_setting():
    """Test that MCP events are set when loading completes."""
    tf = ToolFactory()

    async def fast_resolver():
        return []

    tf.add_mcp("test_server", fast_resolver)
    # Test that we can await the server to load
    await tf.await_mcp_loaded("test_server")


@pytest.mark.asyncio
async def test_tool_factory_registry_immutability():
    """Test that the registry is not shared between instances."""
    tf1 = ToolFactory()
    tf2 = ToolFactory()

    tf1.add("test", lambda: "test1")
    tf2.add("test", lambda: "test2")

    # Test that each instance maintains its own tools
    result1 = await tf1.resolve("test")()
    result2 = await tf2.resolve("test")()
    assert result1 == "test1"
    assert result2 == "test2"


@pytest.mark.asyncio
async def test_tool_factory_mcp_namespace_isolation():
    """Test that MCP namespaces are isolated between instances."""
    tf1 = ToolFactory()
    tf2 = ToolFactory()

    async def resolver1():
        return []

    async def resolver2():
        return []

    tf1.add_mcp("server", resolver1)
    tf2.add_mcp("server", resolver2)

    # Test that each instance can await its own server independently
    await tf1.await_mcp_loaded("server")
    await tf2.await_mcp_loaded("server")


@pytest.mark.asyncio
async def test_langchain_tool_wrapper_input_formats(
    mcp_server_config: dict[str, dict[str, object]],
) -> None:
    """Test that LangChain tool wrapper handles different input formats correctly."""
    tf = ToolFactory()
    tf.add_mcp("test_server", mcp_config=mcp_server_config)
    await tf.await_mcp_loaded("test_server")

    add_tool = tf.resolve("mcp:test_server:add")

    # Test with dict input (as used in existing tests)
    result1 = await add_tool.ainvoke({"a": 2, "b": 3})  # type: ignore
    assert int(result1) == 5

    # Test with kwargs input (converted to dict)
    result2 = await add_tool.ainvoke({"a": 4, "b": 5})  # type: ignore
    assert int(result2) == 9

    # Test with positional args (should be wrapped in dict)
    result3 = await add_tool.ainvoke({"a": 1, "b": 2})  # type: ignore
    assert int(result3) == 3

    multiply_tool = tf.resolve("mcp:test_server:multiply")
    result4 = await multiply_tool.ainvoke({"a": 3, "b": 4})  # type: ignore
    assert int(result4) == 12
