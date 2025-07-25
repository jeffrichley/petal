import asyncio
from typing import Any

import pytest

from petal.core.tool_factory import ToolFactory

# Removed tests that only check registering raw functions or lambdas.


def test_resolve_missing_tool_raises() -> None:
    tf = ToolFactory()
    with pytest.raises(KeyError):
        tf.resolve("not_found")


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


# MCP/namespace/registry tests and other valid tests remain unchanged.
