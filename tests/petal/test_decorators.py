"""
Tests for the @petaltool decorator to ensure LangChain compatibility.
"""

import asyncio
import unittest  # For mock.ANY
from unittest.mock import Mock, patch

import pytest
from langchain_core.tools import BaseTool
from petal.core.decorators import petalmcp, petalmcp_tool, petaltool
from petal.core.registry import ToolRegistry
from petal.core.tool_factory import ToolFactory
from pydantic import BaseModel


class TestPetaltoolDecorator:
    """Test that @petaltool creates LangChain-compatible tools."""

    def test_petaltool_basic_usage(self):
        """Test basic @petaltool usage creates a valid BaseTool."""

        @petaltool
        def search_api(query: str) -> str:
            """Search the API for the given query."""
            return f"Search result for: {query}"

        # Verify it's a BaseTool
        assert isinstance(search_api, BaseTool)
        assert search_api.name == "search_api"
        assert "Search the API" in search_api.description

        # Test that it can be invoked
        result = search_api.run({"query": "test"})
        assert "Search result for: test" in result

    def test_petaltool_with_custom_name(self):
        """Test @petaltool with custom name."""

        @petaltool("custom_search")
        def search_api(query: str) -> str:
            """Search the API for the given query."""
            return f"Search result for: {query}"

        # Verify custom name is used
        assert isinstance(search_api, BaseTool)
        assert search_api.name == "custom_search"

    def test_petaltool_preserves_metadata(self):
        """Test that @petaltool preserves function metadata for LangChain compatibility."""

        @petaltool
        def test_tool(query: str) -> str:
            """A test tool with metadata."""
            return f"Processed: {query}"

        # Verify metadata is preserved
        assert hasattr(test_tool, "_petal_registered")
        assert test_tool._petal_registered is True
        assert hasattr(test_tool, "_original_func")
        assert test_tool._original_func.__name__ == "test_tool"

    def test_petaltool_with_schema(self):
        """Test @petaltool with custom schema."""

        class SearchSchema(BaseModel):
            query: str
            limit: int = 10

        @petaltool(args_schema=SearchSchema)
        def search_api(query: str, limit: int = 10) -> str:
            """Search the API for the given query."""
            return f"Search result for: {query} (limit: {limit})"

        # Verify schema is set
        assert isinstance(search_api, BaseTool)
        assert search_api.args_schema is not None

    def test_petaltool_with_parse_docstring(self):
        """Test @petaltool with parse_docstring=True."""

        @petaltool(parse_docstring=True)
        def search_api(query: str, limit: int = 10) -> str:
            """Search the API for the given query with limit.

            Args:
                query: The search query to execute.
                limit: Maximum number of results to return.
            """
            return f"Search result for: {query} (limit: {limit})"

        # Verify docstring parsing works
        assert isinstance(search_api, BaseTool)
        assert "Search the API for the given query with limit" in search_api.description

    @pytest.mark.asyncio
    async def test_petaltool_in_langchain_agent(self):
        """Test that @petaltool tools work in LangChain agents."""

        @petaltool
        def calculator(expression: str) -> str:
            """Calculate the result of a mathematical expression."""
            try:
                return str(eval(expression))
            except Exception as e:
                return f"Error: {e}"

        @petaltool
        def echo(message: str) -> str:
            """Echo back the input message."""
            return f"Echo: {message}"

        # Test that tools are valid BaseTool instances
        assert isinstance(calculator, BaseTool)
        assert isinstance(echo, BaseTool)

        # Test that tools have the required attributes for LangChain agents
        assert hasattr(calculator, "name")
        assert hasattr(calculator, "description")
        assert hasattr(calculator, "args_schema")
        assert hasattr(calculator, "run")
        assert hasattr(calculator, "arun")

        # Test that tools can be invoked directly
        calc_result = await calculator.ainvoke({"expression": "2 + 2"})
        assert "4" in calc_result

        echo_result = await echo.ainvoke({"message": "test"})
        assert "Echo: test" in echo_result

        # Test that tools work with LangChain's tool binding

        # Create a mock LLM that supports tool calling
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)

        # Verify that our tools can be bound to the LLM
        bound_llm = mock_llm.bind_tools([calculator, echo])
        assert bound_llm == mock_llm
        mock_llm.bind_tools.assert_called_once_with([calculator, echo])

    def test_petaltool_function_signature_preservation(self):
        """Test that @petaltool preserves function signatures."""

        @petaltool
        def test_func(x: str, y: int = 10) -> str:
            """Test function with signature."""
            return f"{x} {y}"

        # Verify the tool is callable with the right signature
        result = test_func.run({"x": "hello", "y": 5})
        assert "hello 5" in result

    def test_petaltool_async_function(self):
        """Test @petaltool with async functions."""

        @petaltool
        async def async_search(query: str) -> str:
            """Async search function."""
            return f"Async search: {query}"

        # Verify async function works
        assert isinstance(async_search, BaseTool)
        assert async_search.name == "async_search"

    def test_petaltool_error_handling(self):
        """Test @petaltool error handling."""

        @petaltool
        def error_tool() -> str:
            """Tool that raises an error."""
            raise ValueError("Test error")

        # Test that errors are properly handled
        with pytest.raises(ValueError, match="Test error"):
            error_tool.run({})

    def test_petaltool_multiple_registration(self):
        """Test that @petaltool tools can be registered multiple times."""

        @petaltool
        def tool1() -> str:
            """First tool."""
            return "tool1"

        @petaltool
        def tool2() -> str:
            """Second tool."""
            return "tool2"

        # Both should be registered
        registry = ToolRegistry()
        assert "tool1" in registry.list()
        assert "tool2" in registry.list()

    def test_petaltool_langchain_tool_equivalence(self):
        """Test that @petaltool creates tools equivalent to LangChain's @tool."""

        from langchain.tools import tool as langchain_tool

        @petaltool
        def petal_tool(query: str) -> str:
            """Petal tool."""
            return f"Petal: {query}"

        @langchain_tool
        def langchain_tool_func(query: str) -> str:
            """LangChain tool."""
            return f"LangChain: {query}"

        # Both should be BaseTool instances
        assert isinstance(petal_tool, BaseTool)
        assert isinstance(langchain_tool_func, BaseTool)

        # Both should have similar attributes
        assert hasattr(petal_tool, "name")
        assert hasattr(petal_tool, "description")
        assert hasattr(petal_tool, "run")

    def test_petaltool_docstring_handling(self):
        """Test @petaltool docstring handling."""

        @petaltool
        def no_docstring_tool(query: str) -> str:
            """A tool with a docstring."""
            return f"Result: {query}"

        @petaltool(description="Custom description")
        def custom_description_tool(query: str) -> str:
            """A tool with custom description."""
            return f"Result: {query}"

        # Verify default behavior (LangChain uses docstring if present)
        assert isinstance(no_docstring_tool, BaseTool)
        assert "A tool with a docstring" in no_docstring_tool.description

        # Verify custom description overrides docstring
        assert isinstance(custom_description_tool, BaseTool)
        assert custom_description_tool.description == "Custom description"

    def test_petaltool_response_format(self):
        """Test @petaltool with different response formats."""

        @petaltool(response_format="content_and_artifact")
        def artifact_tool(query: str) -> tuple[str, dict]:
            """Tool that returns content and artifact."""
            return f"Result: {query}", {"metadata": "test"}

        # Verify response format is set
        assert isinstance(artifact_tool, BaseTool)
        assert artifact_tool.response_format == "content_and_artifact"

    def test_petaltool_infer_schema_disabled(self):
        """Test @petaltool with infer_schema=False."""

        @petaltool(infer_schema=False)
        def simple_tool(query: str) -> str:
            """A simple tool without schema inference."""
            return f"Simple: {query}"

        # Verify schema inference is disabled
        assert isinstance(simple_tool, BaseTool)
        assert simple_tool.args_schema is None

        # Verify it still works
        result = simple_tool.run({"query": "test"})
        assert "Simple: test" in result

    @pytest.mark.asyncio
    async def test_petaltool_direct_invocation_methods(self):
        """Test that @petaltool tools can be invoked directly with .run() and .ainvoke()."""

        @petaltool
        def sync_tool(query: str) -> str:
            """A synchronous tool for testing direct invocation."""
            return f"Sync result: {query}"

        @petaltool
        async def async_tool(query: str) -> str:
            """An asynchronous tool for testing direct invocation."""
            return f"Async result: {query}"

        # Test synchronous tool with .run()
        sync_result = sync_tool.run({"query": "test_sync"})
        assert "Sync result: test_sync" in sync_result

        # Test synchronous tool with .ainvoke()
        sync_async_result = await sync_tool.ainvoke({"query": "test_sync_async"})
        assert "Sync result: test_sync_async" in sync_async_result

        # Test asynchronous tool with .ainvoke()
        async_result = await async_tool.ainvoke({"query": "test_async"})
        assert "Async result: test_async" in async_result

        # Test that .run() on async tool raises NotImplementedError
        with pytest.raises(NotImplementedError):
            async_tool.run({"query": "test"})

    def test_petaltool_with_complex_arguments(self):
        """Test @petaltool with complex argument types."""

        @petaltool
        def complex_tool(text: str, count: int = 1) -> str:
            """Tool with multiple arguments."""
            return f"Processed: {text} (count: {count})"

        # Test with multiple arguments
        result = complex_tool.run({"text": "hello", "count": 3})
        assert "Processed: hello (count: 3)" in result

    @pytest.mark.asyncio
    async def test_petaltool_error_handling_in_async_context(self):
        """Test @petaltool error handling in async context."""

        @petaltool
        async def async_error_tool() -> str:
            """Async tool that raises an error."""
            raise RuntimeError("Async error")

        # Test that async errors are properly handled
        with pytest.raises(RuntimeError, match="Async error"):
            await async_error_tool.ainvoke({})

        # Test that .run() on async tool raises NotImplementedError
        with pytest.raises(NotImplementedError):
            async_error_tool.run({})

    def test_petaltool_always_returns_basetool(self):
        """Test that @petaltool always returns a BaseTool or derivative."""

        # Test basic usage
        @petaltool
        def basic_tool(query: str) -> str:
            """Basic tool."""
            return f"Result: {query}"

        assert isinstance(basic_tool, BaseTool)

        # Test with custom name
        @petaltool("custom_name")
        def named_tool(query: str) -> str:
            """Named tool."""
            return f"Result: {query}"

        assert isinstance(named_tool, BaseTool)

        # Test with all parameters
        @petaltool(
            description="Custom description",
            return_direct=True,
            infer_schema=False,
            response_format="content",
        )
        def complex_tool(query: str) -> str:
            """Complex tool."""
            return f"Result: {query}"

        assert isinstance(complex_tool, BaseTool)

        # Test async function
        @petaltool
        async def async_tool(query: str) -> str:
            """Async tool."""
            return f"Result: {query}"

        assert isinstance(async_tool, BaseTool)

        # Test function with no docstring
        @petaltool(description="A tool without a docstring")
        def no_docstring_tool(query: str) -> str:
            return f"Result: {query}"

        assert isinstance(no_docstring_tool, BaseTool)

        # Verify all tools have BaseTool methods
        for tool_obj in [
            basic_tool,
            named_tool,
            complex_tool,
            async_tool,
            no_docstring_tool,
        ]:
            assert hasattr(tool_obj, "run")
            assert hasattr(tool_obj, "arun")
            assert hasattr(tool_obj, "name")
            assert hasattr(tool_obj, "description")
            assert hasattr(tool_obj, "args_schema")


class TestPetalMCPDecorators:
    def test_petalmcp_registers_server_with_tool_factory(self):
        """@petalmcp should call ToolFactory.add_mcp with correct args."""
        with patch.object(ToolFactory, "add_mcp", autospec=True) as mock_add_mcp:
            config = {
                "url": {"endpoint": "http://localhost:8000/mcp", "meta": 123}
            }  # object value for mypy

            @petalmcp("test_server", config=config)
            class TestServer:
                pass

            mock_add_mcp.assert_called_once_with(
                unittest.mock.ANY, "test_server", mcp_config=config
            )

    def test_petalmcp_tool_registers_function_with_tool_factory(self):
        """@petalmcp_tool should call ToolFactory.add with correct namespace."""
        with patch.object(ToolFactory, "add", autospec=True) as mock_add:

            @petalmcp_tool("mcp:myserver:mytool")
            def mytool(x: int) -> int:
                return x + 1

            # Should register under the correct name
            args, kwargs = mock_add.call_args
            assert args[1] == "mcp:myserver:mytool"
            assert callable(args[2])

    @pytest.mark.asyncio
    async def test_petalmcp_decorator_uses_official_mcp_client(self):
        """@petalmcp should not create custom MCP client, but use ToolFactory.add_mcp (which uses MultiServerMCPClient)."""
        with patch("langchain_mcp_adapters.client.MultiServerMCPClient", autospec=True):
            config = {
                "url": {"endpoint": "http://localhost:8000/mcp", "meta": 123}
            }  # object value for mypy
            tf = ToolFactory()
            tf.add_mcp("test_server", mcp_config=config)
            # Wait for the MCP loading to complete to avoid the warning
            await tf.await_mcp_loaded("test_server")

    def test_petalmcp_tool_duplicate_name_raises(self):
        """@petalmcp_tool should raise if the tool name is already registered."""
        tf = ToolFactory()

        @petaltool
        def dupe_tool(x: int) -> int:
            """Duplicate tool for testing."""
            return x

        tf.add("mcp:dupe:tool", dupe_tool)
        with (
            patch.object(ToolFactory, "add", side_effect=KeyError("already exists")),
            pytest.raises(KeyError),
        ):

            @petalmcp_tool("mcp:dupe:tool")
            def dupe(x: int) -> int:
                return x


class TestToolFactoryErrorCases:
    """Test error cases in ToolFactory.add() and ToolFactory.resolve() methods."""

    def test_tool_factory_add_with_non_basetool_raises_typeerror(self):
        """ToolFactory.add() should raise TypeError when adding non-BaseTool objects."""
        tf = ToolFactory()

        # Test with a regular function (not decorated with @petaltool)
        def regular_function(x: int) -> int:
            return x + 1

        with pytest.raises(
            TypeError,
            match="Tool 'test_tool' must be decorated with @tool or @petaltool",
        ):
            tf.add("test_tool", regular_function)  # type: ignore[arg-type]

        # Test with a string
        with pytest.raises(
            TypeError,
            match="Tool 'test_tool' must be decorated with @tool or @petaltool",
        ):
            tf.add("test_tool", "not a tool")  # type: ignore[arg-type]

        # Test with None
        with pytest.raises(
            TypeError,
            match="Tool 'test_tool' must be decorated with @tool or @petaltool",
        ):
            tf.add("test_tool", None)  # type: ignore[arg-type]

        # Test with an integer
        with pytest.raises(
            TypeError,
            match="Tool 'test_tool' must be decorated with @tool or @petaltool",
        ):
            tf.add("test_tool", 42)  # type: ignore[arg-type]

    def test_tool_factory_resolve_nonexistent_tool_raises_keyerror(self):
        """ToolFactory.resolve() should raise KeyError for non-existent tools."""
        tf = ToolFactory()

        with pytest.raises(
            KeyError, match="Tool 'nonexistent_tool' not found in registry."
        ):
            tf.resolve("nonexistent_tool")

    @pytest.mark.asyncio
    async def test_tool_factory_resolve_mcp_tool_still_loading_raises_keyerror(self):
        tf = ToolFactory()
        event = asyncio.Event()
        tf._mcp_loaded["mcp:test_server"] = event
        # Set the event so the coroutine completes and does not hang
        event.set()
        with pytest.raises(
            KeyError, match="MCP tool 'mcp:test_server:tool' is still loading"
        ):
            tf.resolve("mcp:test_server:tool")

    def test_tool_factory_resolve_mcp_tool_server_not_registered_raises_keyerror(self):
        """ToolFactory.resolve() should raise KeyError for MCP tools with unregistered server."""
        tf = ToolFactory()

        with pytest.raises(
            KeyError,
            match="MCP tool 'mcp:unknown_server:tool' not found. Server 'unknown_server' has not been registered.",
        ):
            tf.resolve("mcp:unknown_server:tool")

    def test_tool_factory_resolve_corrupted_registry_raises_typeerror(self):
        """ToolFactory.resolve() should raise TypeError if registry contains non-BaseTool objects."""
        tf = ToolFactory()

        # Manually corrupt the registry with a non-BaseTool object
        tf._registry["corrupted_tool"] = "not a BaseTool"  # type: ignore[assignment]

        with pytest.raises(
            TypeError, match="Tool 'corrupted_tool' is not a BaseTool instance"
        ):
            tf.resolve("corrupted_tool")

    def test_tool_factory_add_mcp_without_config_raises_valueerror(self):
        """ToolFactory.add_mcp() should raise ValueError when called without config or resolver."""
        tf = ToolFactory()

        with pytest.raises(
            ValueError, match="mcp_config is required when resolver is None"
        ):
            tf.add_mcp("test_server")

    def test_tool_factory_add_mcp_with_invalid_mcp_tool_name_raises_keyerror(self):
        """ToolFactory.resolve() should handle MCP tools with invalid name format."""
        tf = ToolFactory()

        # Test with MCP tool name that doesn't have enough parts
        with pytest.raises(KeyError, match="Tool 'mcp:invalid' not found in registry."):
            tf.resolve("mcp:invalid")

        # Test with MCP tool name that has too many parts but server not registered
        with pytest.raises(
            KeyError,
            match="MCP tool 'mcp:test:tool:extra' not found. Server 'test' has not been registered.",
        ):
            tf.resolve("mcp:test:tool:extra")

    @pytest.mark.asyncio
    async def test_tool_factory_chaining_works_correctly(self, mcp_server_config):
        tf = ToolFactory()

        @petaltool
        def test_tool(x: int) -> int:
            """Test tool for chaining."""
            return x + 1

        result = tf.add("test_tool", test_tool)
        assert result is tf
        # Use the real MCP server fixture instead of mocking
        result = tf.add_mcp("test_server", mcp_config=mcp_server_config)
        assert result is tf
        await tf.await_mcp_loaded("test_server")

        # Verify that the MCP tools were loaded
        tool_names = tf.list()
        assert "mcp:test_server:add" in tool_names
        assert "mcp:test_server:multiply" in tool_names

    def test_tool_factory_list_returns_sorted_names(self):
        """ToolFactory.list() should return sorted list of tool names."""
        tf = ToolFactory()

        @petaltool
        def tool_c(x: int) -> int:
            """Tool C for testing."""
            return x

        @petaltool
        def tool_a(x: int) -> int:
            """Tool A for testing."""
            return x

        @petaltool
        def tool_b(x: int) -> int:
            """Tool B for testing."""
            return x

        tf.add("tool_c", tool_c)
        tf.add("tool_a", tool_a)
        tf.add("tool_b", tool_b)

        tool_list = tf.list()
        assert tool_list == ["tool_a", "tool_b", "tool_c"]
