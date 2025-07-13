"""
Tests for the @petaltool decorator to ensure LangChain compatibility.
"""

from unittest.mock import Mock

import pytest
from langchain_core.tools import BaseTool
from petal.core.decorators import petaltool
from petal.core.registry import ToolRegistry


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

        from pydantic import BaseModel

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
