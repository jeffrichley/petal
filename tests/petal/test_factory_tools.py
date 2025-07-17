import pytest
from langchain_core.messages import AIMessage
from petal.core.factory import AgentFactory

from tests.petal.conftest_factory import ChatState


@pytest.mark.asyncio
async def test_with_react_tools():
    """Test with_react_tools() with scratchpad support."""
    from langchain_core.tools import tool

    @tool
    def react_tool(query: str) -> str:
        """A tool for ReAct testing."""
        return f"ReAct: {query}"

    agent = await (
        AgentFactory(ChatState)
        .with_chat()
        .with_react_tools([react_tool], scratchpad_key="observations")
        .build()
    )

    # Verify the agent was built successfully
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_with_react_tools_default_scratchpad():
    """Test with_react_tools() with default scratchpad key."""
    from langchain_core.tools import tool

    @tool
    def default_tool(query: str) -> str:
        """A tool for default scratchpad testing."""
        return f"Default: {query}"

    agent = await (
        AgentFactory(ChatState)
        .with_chat()
        .with_react_tools([default_tool])  # Should use default "scratchpad" key
        .build()
    )

    # Verify the agent was built successfully
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_with_tools_no_scratchpad():
    """Test with_tools() without scratchpad (basic tool support)."""
    from langchain_core.tools import tool

    @tool
    def basic_tool(query: str) -> str:
        """A basic tool without scratchpad."""
        return f"Basic: {query}"

    agent = await (
        AgentFactory(ChatState)
        .with_chat()
        .with_tools([basic_tool])  # No scratchpad_key specified
        .build()
    )

    # Verify the agent was built successfully
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_with_tools_custom_scratchpad():
    """Test with_tools() with custom scratchpad key."""
    from langchain_core.tools import tool

    @tool
    def custom_tool(query: str) -> str:
        """A tool with custom scratchpad."""
        return f"Custom: {query}"

    agent = await (
        AgentFactory(ChatState)
        .with_chat()
        .with_tools([custom_tool], scratchpad_key="custom_observations")
        .build()
    )

    # Verify the agent was built successfully
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_with_tools_fluent_chaining():
    """Test that with_tools() supports fluent chaining."""
    from langchain_core.tools import tool

    @tool
    def chained_tool(query: str) -> str:
        """A tool for chaining test."""
        return f"Chained: {query}"

    factory = AgentFactory(ChatState)
    result = factory.with_chat().with_tools([chained_tool])

    # Verify fluent chaining returns self
    assert result is factory

    # Verify the agent can still be built
    agent = await factory.build()
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_with_react_tools_fluent_chaining():
    """Test that with_react_tools() supports fluent chaining."""
    from langchain_core.tools import tool

    @tool
    def react_chained_tool(query: str) -> str:
        """A tool for ReAct chaining test."""
        return f"ReAct Chained: {query}"

    factory = AgentFactory(ChatState)
    result = factory.with_chat().with_react_tools([react_chained_tool])

    # Verify fluent chaining returns self
    assert result is factory

    # Verify the agent can still be built
    agent = await factory.build()
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_tool_factory_integration():
    """Test that AgentFactory properly integrates with ToolRegistry."""
    from langchain_core.tools import tool

    @tool
    def factory_tool(query: str) -> str:
        """A tool for factory integration testing."""
        return f"Factory: {query}"

    # Create factory and register tool using ToolRegistry singleton
    factory = AgentFactory(ChatState)
    from petal.core.registry import ToolRegistry

    registry = ToolRegistry()
    registry.add("factory_tool", factory_tool)

    # Use string name - should resolve via ToolRegistry
    agent = await factory.with_chat().with_tools(["factory_tool"]).build()

    # Verify the agent was built successfully
    assert agent is not None
    assert agent.built is True


@pytest.mark.asyncio
async def test_tool_factory_resolution_error():
    """Test that AgentFactory handles ToolFactory resolution errors gracefully."""
    # Try to use a tool name that doesn't exist
    with pytest.raises(KeyError, match="Tool 'nonexistent_tool' not found"):
        await (
            AgentFactory(ChatState).with_chat().with_tools(["nonexistent_tool"]).build()
        )


@pytest.mark.asyncio
async def test_with_tools_empty_list():
    """Test with_tools() with empty tool list raises error."""
    with pytest.raises(
        ValueError, match="Tools list cannot be empty. Provide at least one tool."
    ):
        await AgentFactory(ChatState).with_chat().with_tools([]).build()


@pytest.mark.asyncio
async def test_with_react_tools_empty_list():
    """Test with_react_tools() with empty tool list raises error."""
    with pytest.raises(
        ValueError, match="Tools list cannot be empty. Provide at least one tool."
    ):
        await AgentFactory(ChatState).with_chat().with_react_tools([]).build()


@pytest.mark.asyncio
async def test_with_tools_before_chat_raises_error():
    """Test that with_tools() before with_chat() raises error."""
    from langchain_core.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """A test tool."""
        return f"Test: {query}"

    with pytest.raises(
        ValueError, match="No steps have been added. Call with_chat\\(\\) first."
    ):
        AgentFactory(ChatState).with_tools([test_tool])


@pytest.mark.asyncio
async def test_with_react_tools_before_chat_raises_error():
    """Test that with_react_tools() before with_chat() raises error."""
    from langchain_core.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """A test tool."""
        return f"Test: {query}"

    with pytest.raises(
        ValueError, match="No steps have been added. Call with_chat\\(\\) first."
    ):
        AgentFactory(ChatState).with_react_tools([test_tool])


@pytest.mark.asyncio
async def test_with_tools_non_llm_step_raises_error():
    """Test that with_tools() when the most recent step is not an LLM step raises error."""
    from langchain_core.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """A test tool."""
        return f"Test: {query}"

    async def custom_step(state):  # noqa: ARG001
        """A custom step that is not an LLM step."""
        return {"custom": "value"}

    # Add a custom step first, then try to add tools
    with pytest.raises(
        ValueError,
        match="The most recent step is not an LLM step. Call with_chat\\(\\) first.",
    ):
        AgentFactory(ChatState).add(custom_step).with_tools([test_tool])


@pytest.mark.asyncio
async def test_with_react_tools_non_llm_step_raises_error():
    """Test that with_react_tools() when the most recent step is not an LLM step raises error."""
    from langchain_core.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """A test tool."""
        return f"Test: {query}"

    async def custom_step(state):  # noqa: ARG001
        """A custom step that is not an LLM step."""
        return {"custom": "value"}

    # Add a custom step first, then try to add tools
    with pytest.raises(
        ValueError,
        match="The most recent step is not an LLM step. Call with_chat\\(\\) first.",
    ):
        AgentFactory(ChatState).add(custom_step).with_react_tools([test_tool])


@pytest.mark.asyncio
async def test_tools_are_injected_and_invoke_tool_message():
    """End-to-end: verify tools are injected, invoked, and ToolMessage is appended."""
    from langchain_core.tools import tool

    # --- Tool as direct object ---
    called = {}

    @tool
    async def echo_tool(query: str) -> str:
        """Echoes the input query for testing tool injection."""
        called["direct"] = query
        return f"Echo: {query}"

    # --- Tool as string name ---
    @tool
    async def string_tool(query: str) -> str:
        """Returns a string with the input query for testing tool injection."""
        called["string"] = query
        return f"String: {query}"

    # Minimal mock LLM
    class DummyLLM:
        def __init__(self):
            self._call_count = 0

        async def ainvoke(self, _, config=None, **kwargs):  # noqa: ARG002

            self._call_count += 1
            if self._call_count == 1:
                return AIMessage(
                    content="I need to call a tool",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "string_tool",
                            "args": {"query": "foo"},
                        }
                    ],
                )
            else:
                return AIMessage(content="Tool execution completed successfully")
