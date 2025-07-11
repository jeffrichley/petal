"""Tests for tool step strategy."""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import tool
from langgraph.graph import END
from petal.core.steps.tool import (
    ToolStep,
    ToolStepStrategy,
    decide_next_step,
    tools_condition,
)


class TestToolStep:
    """Test ToolStep class."""

    @pytest.fixture
    def mock_tool(self):
        """Create a proper LangChain tool."""

        @tool
        def test_tool(query: str) -> str:
            """A test tool that returns the query."""
            return f"Result: {query}"

        return test_tool

    @pytest.fixture
    def mock_tool_node(self):
        """Create a mock tool node."""
        tool_node = AsyncMock()
        tool_node.ainvoke.return_value = {
            "messages": [
                MagicMock(content="Tool result 1"),
                MagicMock(content="Tool result 2"),
            ]
        }
        return tool_node

    @pytest.fixture
    def tool_step(self, mock_tool):
        """Create a ToolStep instance."""
        return ToolStep([mock_tool])

    def test_tool_step_initialization(self, mock_tool):
        """Test ToolStep initialization."""
        tool_step = ToolStep([mock_tool])
        assert tool_step.tools == [mock_tool]
        assert tool_step.scratchpad_key is None

    def test_tool_step_initialization_with_scratchpad(self, mock_tool):
        """Test ToolStep initialization with scratchpad key."""
        tool_step = ToolStep([mock_tool], scratchpad_key="scratchpad")
        assert tool_step.tools == [mock_tool]
        assert tool_step.scratchpad_key == "scratchpad"

    @pytest.mark.asyncio
    async def test_tool_step_execution(self, tool_step, mock_tool_node, monkeypatch):
        """Test tool step execution."""
        # Mock the ToolNode
        monkeypatch.setattr(tool_step, "_tool_node", mock_tool_node)

        state = {"messages": [MagicMock()]}
        result = await tool_step(state)

        mock_tool_node.ainvoke.assert_called_once_with(state)
        assert "messages" in result

    @pytest.mark.asyncio
    async def test_tool_step_execution_with_scratchpad(
        self, mock_tool, mock_tool_node, monkeypatch
    ):
        """Test tool step execution with scratchpad."""
        tool_step = ToolStep([mock_tool], scratchpad_key="scratchpad")
        monkeypatch.setattr(tool_step, "_tool_node", mock_tool_node)

        state = {"messages": [MagicMock()], "scratchpad": "Previous thoughts"}
        result = await tool_step(state)

        assert (
            result["scratchpad"]
            == "Previous thoughts\nObservation: Tool result 1\nObservation: Tool result 2"
        )

    @pytest.mark.asyncio
    async def test_tool_step_execution_empty_scratchpad(
        self, mock_tool, mock_tool_node, monkeypatch
    ):
        """Test tool step execution with empty scratchpad."""
        tool_step = ToolStep([mock_tool], scratchpad_key="scratchpad")
        monkeypatch.setattr(tool_step, "_tool_node", mock_tool_node)

        state = {"messages": [MagicMock()]}
        result = await tool_step(state)

        assert (
            result["scratchpad"]
            == "Observation: Tool result 1\nObservation: Tool result 2"
        )

    @pytest.mark.asyncio
    async def test_tool_step_execution_no_tool_messages(
        self, mock_tool, mock_tool_node, monkeypatch
    ):
        """Test tool step execution with no tool messages."""
        tool_step = ToolStep([mock_tool], scratchpad_key="scratchpad")
        mock_tool_node.ainvoke.return_value = {"messages": []}
        monkeypatch.setattr(tool_step, "_tool_node", mock_tool_node)

        state = {"messages": [MagicMock()], "scratchpad": "Previous thoughts"}
        result = await tool_step(state)

        assert result["scratchpad"] == "Previous thoughts"

    @pytest.mark.asyncio
    async def test_tool_step_execution_no_scratchpad_key(
        self, mock_tool, mock_tool_node, monkeypatch
    ):
        """Test tool step execution without scratchpad key."""
        tool_step = ToolStep([mock_tool])
        monkeypatch.setattr(tool_step, "_tool_node", mock_tool_node)

        state = {"messages": [MagicMock()]}
        result = await tool_step(state)

        assert "scratchpad" not in result


class TestToolStepStrategy:
    """Test ToolStepStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Create a ToolStepStrategy instance."""
        return ToolStepStrategy()

    @pytest.fixture
    def mock_tool(self):
        """Create a proper LangChain tool."""

        @tool
        def test_tool(query: str) -> str:
            """A test tool that returns the query."""
            return f"Result: {query}"

        return test_tool

    def test_create_step_with_tools(self, strategy, mock_tool):
        """Test creating a step with tools."""
        config: Dict[str, Any] = {"tools": [mock_tool]}
        step = strategy.create_step(config)

        assert isinstance(step, ToolStep)
        assert step.tools == [mock_tool]
        assert step.scratchpad_key is None

    def test_create_step_with_scratchpad(self, strategy, mock_tool):
        """Test creating a step with scratchpad key."""
        config: Dict[str, Any] = {"tools": [mock_tool], "scratchpad_key": "scratchpad"}
        step = strategy.create_step(config)

        assert isinstance(step, ToolStep)
        assert step.tools == [mock_tool]
        assert step.scratchpad_key == "scratchpad"

    def test_create_step_no_tools(self, strategy):
        """Test creating a step without tools raises error."""
        config: Dict[str, Any] = {"tools": []}

        with pytest.raises(ValueError, match="Tool step requires at least one tool"):
            strategy.create_step(config)

    def test_create_step_missing_tools(self, strategy):
        """Test creating a step with missing tools raises error."""
        config: Dict[str, Any] = {}

        with pytest.raises(ValueError, match="Tool step requires at least one tool"):
            strategy.create_step(config)

    def test_get_node_name(self, strategy):
        """Test node name generation."""
        assert strategy.get_node_name(0) == "tool_step_0"
        assert strategy.get_node_name(1) == "tool_step_1"
        assert strategy.get_node_name(42) == "tool_step_42"


class TestDecideNextStep:
    """Test decide_next_step function."""

    def test_decide_next_step_with_tool_calls(self):
        """Test decide_next_step when last message has tool calls."""
        mock_message = MagicMock()
        mock_message.tool_calls = [MagicMock()]

        state: Dict[str, Any] = {"messages": [mock_message]}
        result = decide_next_step(state)

        assert result == "tools"

    def test_decide_next_step_no_tool_calls(self):
        """Test decide_next_step when last message has no tool calls."""
        mock_message = MagicMock()
        mock_message.tool_calls = None

        state: Dict[str, Any] = {"messages": [mock_message]}
        result = decide_next_step(state)

        assert result == END

    def test_decide_next_step_empty_messages(self):
        """Test decide_next_step with empty messages."""
        state: Dict[str, Any] = {"messages": []}
        result = decide_next_step(state)

        assert result == END

    def test_decide_next_step_no_messages_key(self):
        """Test decide_next_step with no messages key."""
        state: Dict[str, Any] = {}
        result = decide_next_step(state)

        assert result == END

    def test_decide_next_step_no_tool_calls_attribute(self):
        """Test decide_next_step when message has no tool_calls attribute."""
        mock_message = MagicMock()
        del mock_message.tool_calls

        state: Dict[str, Any] = {"messages": [mock_message]}
        result = decide_next_step(state)

        assert result == END


class TestToolsCondition:
    """Test tools_condition function."""

    def test_tools_condition_with_tool_calls(self):
        """Test tools_condition when last message has tool calls."""
        mock_message = MagicMock()
        mock_message.tool_calls = [MagicMock()]

        state = {"messages": [mock_message]}
        result = tools_condition(state)

        assert result == "tools"

    def test_tools_condition_no_tool_calls(self):
        """Test tools_condition when last message has no tool calls."""
        mock_message = MagicMock()
        mock_message.tool_calls = None

        state = {"messages": [mock_message]}
        result = tools_condition(state)

        assert result == END

    def test_tools_condition_empty_messages(self):
        """Test tools_condition with empty messages."""
        state: Dict[str, Any] = {"messages": []}
        result = tools_condition(state)

        assert result == END
