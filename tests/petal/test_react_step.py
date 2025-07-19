"""Tests for React step strategy."""

from typing import List
from unittest.mock import AsyncMock, Mock

import pytest
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
        tool_factory.add("test_react_step:test_tool", test_tool)

        config = {
            "tools": ["test_react_step:test_tool"],
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
        tool_factory.add("test_react_step:test_tool1", test_tool1)
        tool_factory.add("test_react_step:test_tool2", test_tool2)

        config = {
            "tools": ["test_react_step:test_tool1", "test_react_step:test_tool2"],
            "state_schema": ReactTestState,
            "llm_instance": Mock(),
            "tool_factory": tool_factory,
        }

        step = await strategy.create_step(config)
        assert callable(step)

    @pytest.mark.asyncio
    async def test_create_step_with_tool_objects(self):
        strategy = ReactStepStrategy()

        @petaltool("custom_tool_1")
        def custom_tool1(query: str) -> str:
            """Test tool for create_step_with_tool_objects."""
            return f"custom_tool1_result: {query}"

        @petaltool("custom_tool_2")
        def custom_tool2(query: str) -> str:
            """Test tool for create_step_with_tool_objects."""
            return f"custom_tool2_result: {query}"

        tool_factory = ToolFactory()
        tool_factory.add("test_react_step:custom_tool_1", custom_tool1)
        tool_factory.add("test_react_step:custom_tool_2", custom_tool2)

        config = {
            "tools": [custom_tool1, custom_tool2],
            "llm": Mock(),
            "max_iterations": 3,
            "tool_factory": tool_factory,
            "state_schema": ReactTestState,
        }

        step = await strategy.create_step(config)
        assert step is not None

    @pytest.mark.asyncio
    async def test_create_step_with_tool_objects_no_names(self):
        strategy = ReactStepStrategy()

        @petaltool
        def custom_tool1(query: str) -> str:
            """Test tool for create_step_with_tool_objects_no_names."""
            return f"custom_tool1_result: {query}"

        @petaltool
        def custom_tool2(query: str) -> str:
            """Test tool for create_step_with_tool_objects_no_names."""
            return f"custom_tool2_result: {query}"

        tool_factory = ToolFactory()
        tool_factory.add("test_react_step:custom_tool1", custom_tool1)
        tool_factory.add("test_react_step:custom_tool2", custom_tool2)

        config = {
            "tools": [custom_tool1, custom_tool2],
            "llm": Mock(),
            "max_iterations": 3,
            "tool_factory": tool_factory,
            "state_schema": ReactTestState,
        }

        step = await strategy.create_step(config)
        assert step is not None

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
        # Register the string tool in the factory under the base name
        tool_factory.add("string_tool", string_tool_func)
        # Don't register object_tool - it will be added automatically when passed as object

        config = {
            "tools": [
                string_tool_name,  # String name - should be resolved from factory
                object_tool_func,  # Tool object - should be added to factory automatically
            ],
            "tool_factory": tool_factory,
            "state_schema": ReactTestState,
        }

        # This should work now with the fixed implementation
        step = await strategy.create_step(config)
        assert step is not None

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
        strategy = ReactStepStrategy()

        @petaltool("single_tool")
        def single_tool_func(query: str) -> str:
            """Test tool for create_step_with_single_tool_object."""
            return f"single_tool_result: {query}"

        tool_factory = ToolFactory()
        tool_factory.add("test_react_step:single_tool", single_tool_func)

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
        strategy = ReactStepStrategy()

        @petaltool("tool0")
        def tool0(query: str) -> str:
            """Test tool for create_step_with_tool_objects_using_generated_names."""
            return f"tool{0}_result: {query}"

        @petaltool("tool1")
        def tool1(query: str) -> str:
            """Test tool for create_step_with_tool_objects_using_generated_names."""
            return f"tool{1}_result: {query}"

        @petaltool("tool2")
        def tool2(query: str) -> str:
            """Test tool for create_step_with_tool_objects_using_generated_names."""
            return f"tool{2}_result: {query}"

        tool_factory = ToolFactory()
        tool_factory.add("test_react_step:tool0", tool0)
        tool_factory.add("test_react_step:tool1", tool1)
        tool_factory.add("test_react_step:tool2", tool2)

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

        @petaltool("test_tool1")
        def test_tool1(query: str) -> str:
            """Test tool 1."""
            return f"Tool 1 result: {query}"

        @petaltool("test_tool2")
        def test_tool2(query: str) -> str:
            """Test tool 2."""
            return f"Tool 2 result: {query}"

        tool_factory = ToolFactory()
        tool_factory.add("test_react_step:test_tool1", test_tool1)
        tool_factory.add("test_react_step:test_tool2", test_tool2)

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
        assert step is not None

    @pytest.mark.asyncio
    async def test_create_step_with_nonexistent_tool(self):
        """Test creating a React step with a tool name that doesn't exist in factory raises error."""
        strategy = ReactStepStrategy()

        tool_factory = ToolFactory()
        # Don't add any tools to the factory

        config = {
            "tools": ["nonexistent_tool"],
            "state_schema": ReactTestState,
            "tool_factory": tool_factory,
        }

        with pytest.raises(
            ValueError, match="Tool 'nonexistent_tool' not found in factory"
        ):
            await strategy.create_step(config)

    @pytest.mark.asyncio
    async def test_create_step_with_tool_object_no_name_attribute(self):
        """Test creating a React step with a tool object that doesn't have a 'name' attribute raises error."""
        strategy = ReactStepStrategy()

        # Create a mock tool object without a 'name' attribute
        class MockToolWithoutName:
            def __call__(self, query: str) -> str:
                return f"mock_result: {query}"

        tool_without_name = MockToolWithoutName()

        config = {
            "tools": [tool_without_name],
            "state_schema": ReactTestState,
        }

        with pytest.raises(
            ValueError, match="Tool object must have a 'name' attribute"
        ):
            await strategy.create_step(config)
