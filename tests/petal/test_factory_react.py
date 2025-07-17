from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from petal.core.factory import AgentFactory
from petal.core.tool_factory import ToolFactory
from pydantic import BaseModel

from tests.petal.conftest_factory import ChatState


@pytest.mark.asyncio
async def test_with_react_loop():
    """Test with_react_loop() adds a React step with tools."""
    from langchain.tools import tool

    @tool
    def react_tool(query: str) -> str:
        """A test tool for React loop."""
        return f"Processed: {query}"

    tool_factory = ToolFactory()
    tool_factory.add("react_tool", react_tool)

    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        # Create a mock LLM instance with proper async ainvoke method
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Test response")]}
        )
        mock_llm_instance.with_structured_output = Mock(
            return_value=mock_structured_llm
        )

        mock_llm_class.return_value = mock_llm_instance

        factory = AgentFactory(ChatState)
        agent = await factory.with_react_loop(
            ["react_tool"], tool_factory=tool_factory
        ).build()
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_react_loop_with_config():
    """Test with_react_loop() with additional configuration."""
    from langchain.tools import tool

    @tool
    def config_tool(query: str) -> str:
        """A test tool for React loop with config."""
        return f"Configured: {query}"

    tool_factory = ToolFactory()
    tool_factory.add("config_tool", config_tool)

    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        # Create a mock LLM instance with proper async ainvoke method
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Test response")]}
        )
        mock_llm_instance.with_structured_output = Mock(
            return_value=mock_structured_llm
        )

        mock_llm_class.return_value = mock_llm_instance

        factory = AgentFactory(ChatState)
        agent = await factory.with_react_loop(
            ["config_tool"],
            tool_factory=tool_factory,
            system_prompt="You are a helpful assistant.",
        ).build()
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_react_loop_fluent_chaining():
    """Test that with_react_loop() supports fluent chaining."""
    from langchain.tools import tool

    @tool
    def chained_tool(query: str) -> str:
        """A test tool for fluent chaining."""
        return f"Chained: {query}"

    tool_factory = ToolFactory()
    tool_factory.add("chained_tool", chained_tool)

    factory = AgentFactory(ChatState)
    result = factory.with_react_loop(["chained_tool"], tool_factory=tool_factory)
    assert result is factory


@pytest.mark.asyncio
async def test_with_react_loop_empty_tools_raises_error():
    """Test with_react_loop() with empty tool list raises error."""
    tool_factory = ToolFactory()
    factory = AgentFactory(ChatState)
    with pytest.raises(ValueError, match="React steps require at least one tool"):
        await factory.with_react_loop([], tool_factory=tool_factory).build()


@pytest.mark.asyncio
async def test_with_react_loop_with_string_tools():
    """Test with_react_loop() with string tool names."""
    tool_factory = ToolFactory()
    from langchain.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """A test tool for string tool names."""
        return f"String: {query}"

    tool_factory.add("test_tool", test_tool)

    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        # Create a mock LLM instance with proper async ainvoke method
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Test response")]}
        )
        mock_llm_instance.with_structured_output = Mock(
            return_value=mock_structured_llm
        )

        mock_llm_class.return_value = mock_llm_instance

        factory = AgentFactory(ChatState)
        agent = await factory.with_react_loop(
            ["test_tool"], tool_factory=tool_factory
        ).build()
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_react_loop_with_structured_output():
    """Test with_react_loop() with structured output model."""
    from langchain.tools import tool

    class TestOutput(BaseModel):
        answer: str
        confidence: float

    @tool
    def structured_tool(query: str) -> str:
        """A test tool for structured output."""
        return f"Structured: {query}"

    tool_factory = ToolFactory()
    tool_factory.add("structured_tool", structured_tool)

    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        # Create a mock LLM instance with proper async ainvoke method
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Test response")]}
        )
        mock_llm_instance.with_structured_output = Mock(
            return_value=mock_structured_llm
        )

        mock_llm_class.return_value = mock_llm_instance

        factory = AgentFactory(ChatState)
        agent = await (
            factory.with_react_loop(["structured_tool"], tool_factory=tool_factory)
            .with_structured_output(TestOutput)
            .build()
        )
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_react_loop_with_prompt_template():
    """Test with_react_loop() with prompt template."""
    from langchain.tools import tool

    @tool
    def prompt_tool(query: str) -> str:
        """A test tool for prompt template."""
        return f"Prompted: {query}"

    tool_factory = ToolFactory()
    tool_factory.add("prompt_tool", prompt_tool)

    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        # Create a mock LLM instance with proper async ainvoke method
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Test response")]}
        )
        mock_llm_instance.with_structured_output = Mock(
            return_value=mock_structured_llm
        )

        mock_llm_class.return_value = mock_llm_instance

        factory = AgentFactory(ChatState)
        agent = await (
            factory.with_react_loop(["prompt_tool"], tool_factory=tool_factory)
            .with_prompt("Answer the question: {input}")
            .build()
        )
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


@pytest.mark.asyncio
async def test_with_react_loop_with_system_prompt():
    """Test with_react_loop() with system prompt."""
    from langchain.tools import tool

    @tool
    def system_tool(query: str) -> str:
        """A test tool for system prompt."""
        return f"System: {query}"

    tool_factory = ToolFactory()
    tool_factory.add("system_tool", system_tool)

    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        # Create a mock LLM instance with proper async ainvoke method
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )
        mock_llm_instance.bind_tools = Mock(return_value=mock_llm_instance)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Test response")]}
        )
        mock_llm_instance.with_structured_output = Mock(
            return_value=mock_structured_llm
        )

        mock_llm_class.return_value = mock_llm_instance

        factory = AgentFactory(ChatState)
        agent = await (
            factory.with_react_loop(["system_tool"], tool_factory=tool_factory)
            .with_system_prompt("You are a helpful assistant.")
            .build()
        )
        result = await agent.arun({"messages": [HumanMessage(content="test")]})
        assert "messages" in result
