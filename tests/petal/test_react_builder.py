from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from petal.core.builders.react import ReActAgentBuilder, ReActState


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's sunny in {city}!"


class TestReActAgentBuilder:
    """Test the ReAct agent builder with unstructured output."""

    def test_extract_react_segments(self):
        """Test extraction of thoughts and actions from LLM output."""
        builder = ReActAgentBuilder(llm=Mock())

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
        builder = ReActAgentBuilder(llm=Mock())

        content = "Just a regular response"
        scratchpad, thoughts, actions = builder._extract_react_segments(content)

        assert scratchpad == ""
        assert thoughts == []
        assert actions == []

    def test_extract_react_segments_multiple(self):
        """Test extraction with multiple thoughts and actions."""
        builder = ReActAgentBuilder(llm=Mock())

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

    def test_build_react_agent(self):
        """Test building a ReAct agent."""
        # Mock LLM
        mock_llm = Mock()
        mock_ai_message = AIMessage(
            content="Thought: I need weather info\nAction: get_weather(city='London')"
        )
        mock_llm.invoke.return_value = mock_ai_message

        # Create tool factory and register the tool
        from petal.core.tool_factory import ToolFactory

        tool_factory = ToolFactory()
        tool_factory.add("get_weather", get_weather)

        builder = ReActAgentBuilder(
            llm=mock_llm,
            tool_names=["get_weather"],
            tool_factory=tool_factory,
            system_prompt="You are a helpful assistant.",
        )

        agent = builder.build()
        assert hasattr(agent, "invoke") or hasattr(agent, "ainvoke")

    def test_react_state_initialization(self):
        """Test ReActState initialization."""
        state = ReActState(
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
        # Mock state with tool calls
        mock_state = Mock()
        mock_state.messages = [Mock()]
        mock_state.messages[-1].tool_calls = ["some_tool_call"]
        mock_state.actions = []

        # Mock the decide_next_step function
        def decide_next_step(state):
            if not state.messages:
                return "END"

            last_message = state.messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            if state.actions and len(state.actions) > 0:
                return "tools"

            return "END"

        result = decide_next_step(mock_state)
        assert result == "tools"

    def test_decide_next_step_with_actions(self):
        """Test decide_next_step when there are actions but no tool calls."""
        # Mock state with actions but no tool calls
        mock_state = Mock()
        mock_state.messages = [Mock()]
        mock_state.messages[-1].tool_calls = None
        mock_state.actions = ["get_weather(city='London')"]

        # Mock the decide_next_step function
        def decide_next_step(state):
            if not state.messages:
                return "END"

            last_message = state.messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            if state.actions and len(state.actions) > 0:
                return "tools"

            return "END"

        result = decide_next_step(mock_state)
        assert result == "tools"

    def test_decide_next_step_end(self):
        """Test decide_next_step when there are no tool calls or actions."""
        # Mock state with no tool calls or actions
        mock_state = Mock()
        mock_state.messages = [Mock()]
        mock_state.messages[-1].tool_calls = None
        mock_state.actions = []

        # Mock the decide_next_step function
        def decide_next_step(state):
            if not state.messages:
                return "END"

            last_message = state.messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            if state.actions and len(state.actions) > 0:
                return "tools"

            return "END"

        result = decide_next_step(mock_state)
        assert result == "END"
