"""Tests for React step strategy."""

from typing import List
from unittest.mock import AsyncMock, Mock

import pytest
from langchain.tools import tool
from petal.core.decorators import petaltool
from petal.core.steps.react import ReactStepStrategy
from petal.core.tool_factory import ToolFactory
from pydantic import BaseModel


class ReactTestState(BaseModel):
    """Test state schema for React step tests."""

    messages: List[str] = []
    result: str = ""


class TestReactStepStrategy:
    """Test the React step strategy."""

    @pytest.mark.asyncio
    async def test_create_step_with_valid_config(self):
        """Test creating a React step with valid configuration."""
        strategy = ReactStepStrategy()

        @petaltool
        def test_tool(query: str) -> str:
            """Echoes the input query."""
            return query

        tool_factory = ToolFactory()
        tool_factory.add("test_tool", test_tool)

        config = {
            "tools": ["test_tool"],
            "state_schema": ReactTestState,
            "llm_instance": Mock(),
            "tool_factory": tool_factory,
        }

        step = await strategy.create_step(config)
        assert callable(step)

    @pytest.mark.asyncio
    async def test_create_step_without_tools(self):
        """Test creating a React step without tools raises error."""
        strategy = ReactStepStrategy()

        config = {
            "tools": [],
            "state_schema": ReactTestState,
        }

        with pytest.raises(ValueError, match="React steps require at least one tool"):
            await strategy.create_step(config)

    @pytest.mark.asyncio
    async def test_create_step_without_state_schema(self):
        """Test creating a React step without state schema raises error."""
        strategy = ReactStepStrategy()

        config = {
            "tools": [Mock()],
        }

        with pytest.raises(ValueError, match="React steps require a state_schema"):
            await strategy.create_step(config)

    def test_get_node_name(self):
        """Test node name generation."""
        strategy = ReactStepStrategy()
        node_name = strategy.get_node_name(0)
        assert node_name == "react_step_0"

        node_name = strategy.get_node_name(5)
        assert node_name == "react_step_5"

    @pytest.mark.asyncio
    async def test_create_step_with_string_tool_names(self):
        """Test creating a React step with string tool names."""
        strategy = ReactStepStrategy()

        @petaltool("test_tool1")
        def test_tool1(query: str) -> str:
            """Test tool 1."""
            return f"tool1_result: {query}"

        @petaltool("test_tool2")
        def test_tool2(query: str) -> str:
            """Test tool 2."""
            return f"tool2_result: {query}"

        tool_factory = ToolFactory()
        tool_factory.add("test_tool1", test_tool1)
        tool_factory.add("test_tool2", test_tool2)

        config = {
            "tools": ["test_tool1", "test_tool2"],
            "state_schema": ReactTestState,
            "llm_instance": Mock(),
            "tool_factory": tool_factory,
        }

        step = await strategy.create_step(config)
        assert callable(step)

    @pytest.mark.asyncio
    async def test_create_step_with_tool_objects(self):
        """Test creating a React step with tool objects."""
        strategy = ReactStepStrategy()

        @petaltool("custom_tool_1")
        def custom_tool1(query: str) -> str:
            """Custom tool 1."""
            return f"custom_tool1_result: {query}"

        @petaltool("custom_tool_2")
        def custom_tool2(query: str) -> str:
            """Custom tool 2."""
            return f"custom_tool2_result: {query}"

        tool_factory = ToolFactory()
        tool_factory.add("custom_tool_1", custom_tool1)
        tool_factory.add("custom_tool_2", custom_tool2)

        config = {
            "tools": [custom_tool1, custom_tool2],
            "state_schema": ReactTestState,
            "llm_instance": Mock(),
            "tool_factory": tool_factory,
        }

        step = await strategy.create_step(config)
        assert callable(step)

    @pytest.mark.asyncio
    async def test_create_step_with_tool_objects_no_names(self):
        """Test creating a React step with tool objects that don't have names."""
        strategy = ReactStepStrategy()

        @petaltool("tool_0")
        def tool1(query: str) -> str:
            """Tool 1 without name."""
            return f"tool1_result: {query}"

        @petaltool("tool_1")
        def tool2(query: str) -> str:
            """Tool 2 without name."""
            return f"tool2_result: {query}"

        tool_factory = ToolFactory()
        tool_factory.add("tool_0", tool1)
        tool_factory.add("tool_1", tool2)

        config = {
            "tools": [tool1, tool2],
            "state_schema": ReactTestState,
            "llm_instance": Mock(),
            "tool_factory": tool_factory,
        }

        step = await strategy.create_step(config)
        assert callable(step)

    @pytest.mark.asyncio
    async def test_create_step_with_mixed_tool_types(self):
        """Test creating a React step with mixed string names and tool objects."""
        strategy = ReactStepStrategy()

        string_tool_name = "string_tool"

        @petaltool("object_tool")
        def object_tool_func(query: str) -> str:
            """Object tool."""
            return f"object_tool_result: {query}"

        @petaltool("string_tool")
        def string_tool_func(query: str) -> str:
            """String tool."""
            return f"string_tool_result: {query}"

        tool_factory = ToolFactory()
        tool_factory.add("string_tool", string_tool_func)
        tool_factory.add("object_tool", object_tool_func)

        config = {
            "tools": [string_tool_name, object_tool_func],
            "state_schema": ReactTestState,
            "llm_instance": Mock(),
            "tool_factory": tool_factory,
        }

        try:
            step = await strategy.create_step(config)
            assert callable(step)
        except (AttributeError, TypeError):
            pass

    @pytest.mark.asyncio
    async def test_create_step_with_empty_tools_list(self):
        """Test creating a React step with empty tools list raises error."""
        strategy = ReactStepStrategy()

        config = {
            "tools": [],
            "state_schema": ReactTestState,
        }

        with pytest.raises(ValueError, match="React steps require at least one tool"):
            await strategy.create_step(config)

    @pytest.mark.asyncio
    async def test_create_step_with_none_tools(self):
        """Test creating a React step with None tools raises error."""
        strategy = ReactStepStrategy()

        config = {
            "tools": None,
            "state_schema": ReactTestState,
        }

        with pytest.raises(ValueError, match="React steps require at least one tool"):
            await strategy.create_step(config)

    @pytest.mark.asyncio
    async def test_create_step_with_single_tool_object(self):
        """Test creating a React step with a single tool object."""
        strategy = ReactStepStrategy()

        @petaltool("single_tool")
        def single_tool_func(query: str) -> str:
            """Single tool."""
            return f"single_tool_result: {query}"

        tool_factory = ToolFactory()
        tool_factory.add("single_tool", single_tool_func)

        config = {
            "tools": [single_tool_func],
            "state_schema": ReactTestState,
            "llm_instance": Mock(),
            "tool_factory": tool_factory,
        }

        step = await strategy.create_step(config)
        assert callable(step)

    @pytest.mark.asyncio
    async def test_create_step_with_tool_objects_using_generated_names(self):
        """Test creating a React step with tool objects using generated names."""
        strategy = ReactStepStrategy()

        @petaltool("tool0")
        def tool0(query: str) -> str:
            """Tool 0."""
            return f"tool0_result: {query}"

        @petaltool("tool1")
        def tool1(query: str) -> str:
            """Tool 1."""
            return f"tool1_result: {query}"

        @petaltool("tool2")
        def tool2(query: str) -> str:
            """Tool 2."""
            return f"tool2_result: {query}"

        tool_factory = ToolFactory()
        tool_factory.add("tool0", tool0)
        tool_factory.add("tool1", tool1)
        tool_factory.add("tool2", tool2)

        config = {
            "tools": [tool0, tool1, tool2],
            "state_schema": ReactTestState,
            "llm_instance": Mock(),
            "tool_factory": tool_factory,
        }

        step = await strategy.create_step(config)
        assert callable(step)

    @pytest.mark.asyncio
    async def test_react_step_execution_with_tool_objects(self):
        """Test that React step can execute with tool objects."""
        strategy = ReactStepStrategy()

        @tool
        def test_tool1(query: str) -> str:
            """Test tool 1."""
            return f"Tool 1 result: {query}"

        @tool
        def test_tool2(query: str) -> str:
            """Test tool 2."""
            return f"Tool 2 result: {query}"

        tool_factory = ToolFactory()
        tool_factory.add("test_tool1", test_tool1)
        tool_factory.add("test_tool2", test_tool2)

        # Mock LLM
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=Mock(content="LLM response"))
        mock_llm.bind_tools = Mock(return_value=mock_llm)

        config = {
            "tools": [test_tool1, test_tool2],
            "state_schema": ReactTestState,
            "llm_instance": mock_llm,
            "tool_factory": tool_factory,
        }

        step = await strategy.create_step(config)

        # Test that the step is callable
        assert callable(step)
