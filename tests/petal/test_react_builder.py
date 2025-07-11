from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END
from pydantic import BaseModel, Field

from petal.core.builders.react import ReActAgentBuilder, ReActLoopState


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's sunny in {city}!"


class ReactTestState(BaseModel):
    """Test state schema for ReAct agent."""

    messages: list = Field(default_factory=list)
    weather_info: str = ""
    user_name: str = "TestUser"
    location: str = "TestLocation"


class TestReActAgentBuilder:
    """Test the ReAct agent builder with unstructured output."""

    def test_extract_react_segments(self):
        """Test extraction of thoughts and actions from LLM output."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        # Test with both thought and action
        content = (
            "Thought: I need to check the weather\nAction: get_weather(city='London')"
        )
        scratchpad, thoughts, actions = builder._extract_react_segments(content)

        assert "Thought: I need to check the weather" in scratchpad
        assert "Action: get_weather(city='London')" in scratchpad
        assert thoughts == ["I need to check the weather"]
        assert actions == ["get_weather(city='London')"]

    def test_extract_react_segments_no_content(self):
        """Test extraction with no thoughts or actions."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        content = "Just a regular response"
        scratchpad, thoughts, actions = builder._extract_react_segments(content)

        assert scratchpad == ""
        assert thoughts == []
        assert actions == []

    def test_extract_react_segments_multiple(self):
        """Test extraction with multiple thoughts and actions."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        content = """
        Thought: First, I need to check the weather
        Action: get_weather(city='London')
        Thought: Now I need to check the time
        Action: get_time()
        """
        scratchpad, thoughts, actions = builder._extract_react_segments(content)

        assert len(thoughts) == 2
        assert len(actions) == 2
        assert "Thought: First, I need to check the weather" in scratchpad
        assert "Action: get_weather(city='London')" in scratchpad
        assert "Thought: Now I need to check the time" in scratchpad
        assert "Action: get_time()" in scratchpad

    @pytest.mark.asyncio
    async def test_build_react_agent(self):
        """Test building a ReAct agent."""
        # Mock LLM
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="Thought: I need weather info\nAction: get_weather(city='London')"
            )
        )
        mock_llm.bind_tools = Mock(return_value=mock_llm)

        # Mock with_structured_output method
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value=ReactTestState(
                messages=[HumanMessage(content="What's the weather?")],
                weather_info="It's sunny in London!",
            )
        )
        mock_llm.with_structured_output = Mock(return_value=mock_structured_llm)

        # Create tool factory and register the tool
        from petal.core.tool_factory import ToolFactory

        tool_factory = ToolFactory()
        tool_factory.add("get_weather", get_weather)

        builder = ReActAgentBuilder(
            state_schema=ReactTestState,
            llm=mock_llm,
            tool_names=["get_weather"],
            tool_factory=tool_factory,
            system_prompt="You are a helpful assistant.",
        )

        agent = builder.build()
        assert callable(agent)

        # Test that the agent can be called
        test_state = ReactTestState(
            messages=[HumanMessage(content="What's the weather?")]
        )
        result = await agent(test_state)
        assert isinstance(result, ReactTestState)

    def test_react_state_initialization(self):
        """Test ReActLoopState initialization."""
        state = ReActLoopState(
            messages=[HumanMessage(content="What's the weather?")],
            scratchpad="",
            thoughts=[],
            actions=[],
        )

        assert len(state.messages) == 1
        assert state.scratchpad == ""
        assert state.thoughts == []
        assert state.actions == []

    def test_decide_next_step_with_tool_calls(self):
        """Test decide_next_step when there are tool calls."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())
        decide_next_step = builder._create_decide_next_step()

        # Mock state with tool calls
        mock_state = Mock()
        mock_state.messages = [Mock()]
        mock_state.messages[-1].tool_calls = ["some_tool_call"]

        result = decide_next_step(mock_state)
        assert result == "tools"

    def test_decide_next_step_end(self):
        """Test decide_next_step when there are no tool calls."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())
        decide_next_step = builder._create_decide_next_step()

        # Mock state with no tool calls
        mock_state = Mock()
        mock_state.messages = [Mock()]
        mock_state.messages[-1].tool_calls = None

        result = decide_next_step(mock_state)
        assert result == END

    def test_user_to_internal_conversion(self):
        """Test conversion from user state to internal state."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        user_state = ReactTestState(
            messages=[HumanMessage(content="What's the weather?")], weather_info=""
        )

        internal_state = builder._user_to_internal(user_state)
        assert isinstance(internal_state, ReActLoopState)
        assert len(internal_state.messages) == 1
        assert internal_state.scratchpad == ""

    def test_format_prompt_safely(self):
        """Test safe prompt formatting."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        template = "Hello {name}, the weather is {weather}!"
        context = {"name": "Alice", "weather": "sunny"}

        result = builder._format_prompt_safely(template, context)
        assert result == "Hello Alice, the weather is sunny!"

    def test_format_prompt_safely_missing_key(self):
        """Test safe prompt formatting with missing keys."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        template = "Hello {name}, the weather is {weather}!"
        context = {"name": "Alice"}  # Missing weather

        result = builder._format_prompt_safely(template, context)
        assert result == "Hello Alice, the weather is {weather}!"
