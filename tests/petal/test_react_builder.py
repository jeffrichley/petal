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

    def test_extract_react_segments_non_string_input(self):
        """Test extraction with non-string input returns empty values."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        # Test with None
        scratchpad, thoughts, actions = builder._extract_react_segments(None)
        assert scratchpad == ""
        assert thoughts == []
        assert actions == []

        # Test with integer
        scratchpad, thoughts, actions = builder._extract_react_segments(123)
        assert scratchpad == ""
        assert thoughts == []
        assert actions == []

        # Test with list
        scratchpad, thoughts, actions = builder._extract_react_segments(
            ["some", "content"]
        )
        assert scratchpad == ""
        assert thoughts == []
        assert actions == []

        # Test with dict
        scratchpad, thoughts, actions = builder._extract_react_segments(
            {"key": "value"}
        )
        assert scratchpad == ""
        assert thoughts == []
        assert actions == []

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

    def test_decide_next_step_no_messages(self):
        """Test decide_next_step when there are no messages."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())
        decide_next_step = builder._create_decide_next_step()

        # Mock state with no messages
        mock_state = Mock()
        mock_state.messages = []

        result = decide_next_step(mock_state)
        assert result == END

    def test_decide_next_step_no_messages_with_next_node(self):
        """Test decide_next_step when there are no messages but next_node is set."""
        builder = ReActAgentBuilder(
            state_schema=ReactTestState, llm=Mock(), next_node="custom_node"
        )
        decide_next_step = builder._create_decide_next_step()

        # Mock state with no messages
        mock_state = Mock()
        mock_state.messages = []

        result = decide_next_step(mock_state)
        assert result == "custom_node"

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

    def test_safe_model_dump_with_dict(self):
        """Test _safe_model_dump with dict input."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        test_dict = {"key1": "value1", "key2": "value2"}
        result = builder._safe_model_dump(test_dict)

        assert result == {"key1": "value1", "key2": "value2"}
        assert result is not test_dict  # Should be a copy

    def test_safe_model_dump_with_pydantic_model(self):
        """Test _safe_model_dump with Pydantic model."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        # Create a simple Pydantic model for testing
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str = "test"
            value: int = 42

        test_model = TestModel()
        result = builder._safe_model_dump(test_model)

        assert result == {"name": "test", "value": 42}

    def test_safe_model_dump_with_regular_object(self):
        """Test _safe_model_dump with regular object (fallback case)."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        # Create a regular object with __dict__
        class RegularObject:
            def __init__(self):
                self.public_attr = "public"
                self._private_attr = "private"
                self.another_attr = 123

        test_obj = RegularObject()
        result = builder._safe_model_dump(test_obj)

        # Should only include non-private attributes
        assert result == {"public_attr": "public", "another_attr": 123}
        assert "_private_attr" not in result

    def test_safe_get_attr_with_dict(self):
        """Test _safe_get_attr with dict input."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        test_dict = {"key1": "value1", "key2": "value2"}

        # Test existing key
        result = builder._safe_get_attr(test_dict, "key1")
        assert result == "value1"

        # Test non-existing key with default
        result = builder._safe_get_attr(test_dict, "missing_key", "default_value")
        assert result == "default_value"

        # Test non-existing key without default
        result = builder._safe_get_attr(test_dict, "missing_key")
        assert result is None

    def test_safe_get_attr_with_object(self):
        """Test _safe_get_attr with object input."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        class TestObject:
            def __init__(self):
                self.public_attr = "public_value"
                self._private_attr = "private_value"

        test_obj = TestObject()

        # Test existing attribute
        result = builder._safe_get_attr(test_obj, "public_attr")
        assert result == "public_value"

        # Test non-existing attribute with default
        result = builder._safe_get_attr(test_obj, "missing_attr", "default_value")
        assert result == "default_value"

        # Test non-existing attribute without default
        result = builder._safe_get_attr(test_obj, "missing_attr")
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_final_state_schema_mismatch_warning(self):
        """Test _synthesize_final_state warns when user state doesn't match schema."""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=ReactTestState())
        mock_llm.with_structured_output = Mock(return_value=mock_structured_llm)

        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=mock_llm)

        # Create internal state
        internal_state = ReActLoopState(
            messages=[HumanMessage(content="test")],
            scratchpad="",
            thoughts=[],
            actions=[],
            context={},
        )

        # Create a completely different type that will definitely trigger the warning
        original_user_state = (
            object()
        )  # Use a plain object that doesn't have dict-like methods

        # Mock console.print to capture the warning
        with pytest.MonkeyPatch().context() as m:
            mock_console_print = Mock()
            m.setattr("petal.core.builders.react.console.print", mock_console_print)

            result = await builder._synthesize_final_state(
                internal_state, original_user_state
            )

            # Verify warning was printed
            mock_console_print.assert_called_once()
            call_args = mock_console_print.call_args[0][0]
            assert "Warning" in call_args
            assert "Original user state type" in call_args
            assert "doesn't match expected schema" in call_args
            assert "ReactTestState" in call_args

            # Verify the method still returns a result
            assert isinstance(result, ReactTestState)

    @pytest.mark.asyncio
    async def test_synthesize_final_state_context_filtering(self):
        """Test _synthesize_final_state filters context properly."""
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=ReactTestState())
        mock_llm.with_structured_output = Mock(return_value=mock_structured_llm)

        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=mock_llm)

        # Create internal state with various context values
        internal_state = ReActLoopState(
            messages=[HumanMessage(content="test")],
            scratchpad="",
            thoughts=[],
            actions=[],
            context={
                "valid_string": "hello",
                "valid_int": 42,
                "valid_float": 3.14,
                "valid_bool": True,
                "valid_list": [1, 2, 3],
                "valid_dict": {"key": "value"},
                "__langgraph_internal": "should_be_filtered",
                "complex_object": object(),  # Should be filtered
                "function": lambda x: x,  # Should be filtered
            },
        )

        original_user_state = ReactTestState()

        # Mock console.print to avoid output during test
        with pytest.MonkeyPatch().context() as m:
            mock_console_print = Mock()
            m.setattr("petal.core.builders.react.console.print", mock_console_print)

            result = await builder._synthesize_final_state(
                internal_state, original_user_state
            )

            # Verify the method was called with a prompt that includes the filtered context
            mock_structured_llm.ainvoke.assert_called_once()
            call_args = mock_structured_llm.ainvoke.call_args[0][0]

            # Check that valid context values are included
            assert "valid_string" in call_args
            assert "valid_int" in call_args
            assert "valid_float" in call_args
            assert "valid_bool" in call_args
            assert "valid_list" in call_args
            assert "valid_dict" in call_args

            # Check that invalid context values are filtered out
            assert "__langgraph_internal" not in call_args
            assert "complex_object" not in call_args
            assert "function" not in call_args

            # Verify the method still returns a result
            assert isinstance(result, ReactTestState)

    @pytest.mark.asyncio
    async def test_tool_node_with_obs_processing(self):
        """Test _create_tool_node method processes tool messages and builds scratchpad."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        # Mock tool node that returns messages with content
        mock_tool_node = AsyncMock()
        mock_tool_node.ainvoke.return_value = {
            "messages": [
                Mock(content="Tool result 1"),
                Mock(content="Tool result 2"),
                Mock(content=""),  # Empty content
                Mock(),  # No content attribute
            ]
        }

        # Create the tool node wrapper
        tool_node_with_obs = builder._create_tool_node(mock_tool_node)

        # Create a state with existing scratchpad
        state = ReActLoopState(
            messages=[HumanMessage(content="test")],
            scratchpad="Previous thought\nPrevious action",
            thoughts=[],
            actions=[],
            context={"key": "value"},
        )

        # Mock config
        config = Mock()

        result = await tool_node_with_obs(state, config)

        # Verify tool node was invoked
        mock_tool_node.ainvoke.assert_called_once_with(state, config)

        # Verify result contains expected data
        assert "messages" in result
        assert "scratchpad" in result
        assert "context" in result

        # Verify scratchpad includes previous content and new observations
        scratchpad_lines = result["scratchpad"].split("\n")
        expected_lines = [
            "Previous thought",
            "Previous action",
            "Observation: Tool result 1",
            "Observation: Tool result 2",
        ]
        for line in expected_lines:
            assert line in scratchpad_lines
        # Optionally, check that the order is correct (if needed)
        # idxs = [scratchpad_lines.index(line) for line in expected_lines]
        # assert idxs == sorted(idxs)

        # Verify context is preserved
        assert result["context"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_tool_node_with_obs_no_observations(self):
        """Test _create_tool_node when no valid observations are found."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        # Mock tool node that returns no valid messages
        mock_tool_node = AsyncMock()
        mock_tool_node.ainvoke.return_value = {
            "messages": [
                Mock(),  # No content attribute
                Mock(content=""),  # Empty content
                Mock(content=123),  # Non-string content
            ]
        }

        # Create the tool node wrapper
        tool_node_with_obs = builder._create_tool_node(mock_tool_node)

        # Create a state with existing scratchpad
        state = ReActLoopState(
            messages=[HumanMessage(content="test")],
            scratchpad="Previous content",
            thoughts=[],
            actions=[],
            context={"key": "value"},
        )

        # Mock config
        config = Mock()

        result = await tool_node_with_obs(state, config)

        # Verify scratchpad remains unchanged except possibly for empty observation lines
        scratchpad_lines = result["scratchpad"].split("\n")
        assert "Previous content" in scratchpad_lines
        # There should be no non-empty observation lines
        for line in scratchpad_lines:
            if line.startswith("Observation:"):
                # Only allow if it's exactly 'Observation:' (empty)
                assert line.strip() == "Observation:"

        # Verify context is preserved
        assert result["context"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_tool_node_with_obs_empty_state(self):
        """Test _create_tool_node with empty state."""
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=Mock())

        # Mock tool node
        mock_tool_node = AsyncMock()
        mock_tool_node.ainvoke.return_value = {
            "messages": [Mock(content="New observation")]
        }

        # Create the tool node wrapper
        tool_node_with_obs = builder._create_tool_node(mock_tool_node)

        # Create empty state
        state = ReActLoopState(
            messages=[], scratchpad="", thoughts=[], actions=[], context={}
        )

        # Mock config
        config = Mock()

        result = await tool_node_with_obs(state, config)

        # Verify scratchpad only contains the new observation
        assert result["scratchpad"] == "Observation: New observation"

        # Verify context is empty
        assert result["context"] == {}

    def test_init_with_llm_directly(self):
        """Test initialization with LLM provided directly."""
        mock_llm = Mock()
        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=mock_llm)

        assert builder.llm is mock_llm
        assert builder.state_schema is ReactTestState

    def test_init_with_llm_config_openai(self):
        """Test initialization with OpenAI LLM config."""
        llm_config = {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "api_key": "test-key",
        }

        # Mock the ChatOpenAI import and class
        with pytest.MonkeyPatch().context() as m:
            mock_chat_openai = Mock()
            m.setattr("langchain_openai.ChatOpenAI", mock_chat_openai)

            builder = ReActAgentBuilder(
                state_schema=ReactTestState, llm_config=llm_config
            )

            # Verify ChatOpenAI was called with correct parameters
            mock_chat_openai.assert_called_once_with(
                model="gpt-3.5-turbo", temperature=0.7, api_key="test-key"
            )
            assert builder.llm is mock_chat_openai.return_value

    def test_init_with_llm_config_unsupported_provider(self):
        """Test initialization with unsupported LLM provider raises error."""
        llm_config = {"provider": "unsupported_provider", "model": "some-model"}

        with pytest.raises(
            ValueError, match="Unsupported LLM provider: unsupported_provider"
        ):
            ReActAgentBuilder(state_schema=ReactTestState, llm_config=llm_config)

    def test_init_without_llm_or_config(self):
        """Test initialization without LLM or config raises error."""
        with pytest.raises(ValueError, match="Must provide either llm or llm_config"):
            ReActAgentBuilder(state_schema=ReactTestState)

    def test_init_with_system_prompt(self):
        """Test initialization with custom system prompt."""
        mock_llm = Mock()
        system_prompt = "You are a specialized weather assistant."

        builder = ReActAgentBuilder(
            state_schema=ReactTestState, llm=mock_llm, system_prompt=system_prompt
        )

        assert system_prompt in builder.system_prompt
        assert "Thought:" in builder.system_prompt
        assert "Action:" in builder.system_prompt

    def test_init_without_system_prompt(self):
        """Test initialization with default system prompt."""
        mock_llm = Mock()

        builder = ReActAgentBuilder(state_schema=ReactTestState, llm=mock_llm)

        assert "You are a helpful assistant." in builder.system_prompt
        assert "Thought:" in builder.system_prompt
        assert "Action:" in builder.system_prompt

    def test_init_with_system_prompt_already_containing_react_instruction(self):
        """Test initialization with system prompt that already contains REACT_INSTRUCTION."""
        mock_llm = Mock()
        # Create a system prompt that already contains the REACT_INSTRUCTION
        system_prompt = "You are a specialized weather assistant.\n\nFor each step, reason about what to do next by writing a 'Thought:' line, then, if needed, an 'Action:' line (with tool and arguments), and after a tool is called, an 'Observation:' line with the result. Repeat until you can answer the user's question."

        builder = ReActAgentBuilder(
            state_schema=ReactTestState, llm=mock_llm, system_prompt=system_prompt
        )

        # The system prompt should contain the custom text and REACT_INSTRUCTION
        assert "You are a specialized weather assistant." in builder.system_prompt
        assert "Thought:" in builder.system_prompt
        assert "Action:" in builder.system_prompt

        # Count occurrences of REACT_INSTRUCTION to ensure it's not duplicated
        thought_count = builder.system_prompt.count("Thought:")
        action_count = builder.system_prompt.count("Action:")
        observation_count = builder.system_prompt.count("Observation:")

        # Should only appear once each
        assert thought_count == 1
        assert action_count == 1
        assert observation_count == 1

        # Now test that the build method doesn't add it again
        # Mock the tool factory and tools
        from petal.core.tool_factory import ToolFactory

        tool_factory = ToolFactory()
        tool_factory.add("get_weather", get_weather)
        builder.tool_names = ["get_weather"]
        builder.tool_factory = tool_factory
        builder.llm.bind_tools = Mock(return_value=builder.llm)

        # Capture the base_prompt before and after the build method
        original_system_prompt = builder.system_prompt

        # Build the agent
        builder.build()

        # The system prompt should remain unchanged
        assert builder.system_prompt == original_system_prompt

    def test_build_react_agent_tool_documentation(self):
        """Test that tool documentation is created correctly for LangChain tools."""
        from petal.core.tool_factory import ToolFactory

        # Create tool factory and register the tool
        tool_factory = ToolFactory()
        tool_factory.add("get_weather", get_weather)

        # Mock LLM
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)

        builder = ReActAgentBuilder(
            state_schema=ReactTestState,
            llm=mock_llm,
            tool_names=["get_weather"],
            tool_factory=tool_factory,
        )

        # Build the agent to trigger tool documentation creation
        builder.build()

        # Verify that the tool was processed correctly
        # LangChain tools should have both name and description attributes
        tool = tool_factory.resolve("get_weather")
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert not hasattr(tool, "__name__")  # LangChain tools don't have __name__

        # The tool documentation should use name and description, not __name__
        assert tool.name == "get_weather"
        assert "Get weather for a given city" in tool.description

    def test_build_react_agent_requires_tools(self):
        """Test that ReAct agents require tools and throw an error when none are provided."""
        # Mock LLM
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)

        # Test with empty tool list
        builder = ReActAgentBuilder(
            state_schema=ReactTestState,
            llm=mock_llm,
            tool_names=[],  # Empty tool list
        )

        with pytest.raises(ValueError, match="ReAct agents require at least one tool"):
            builder.build()

        # Test with None tool list
        builder = ReActAgentBuilder(
            state_schema=ReactTestState,
            llm=mock_llm,
            tool_names=None,  # None tool list
        )

        with pytest.raises(ValueError, match="ReAct agents require at least one tool"):
            builder.build()
