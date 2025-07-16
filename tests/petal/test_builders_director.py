"""Tests for AgentBuilderDirector."""

from typing import TypedDict

import pytest
from langgraph.graph.message import add_messages
from petal.core.builders.director import AgentBuilderDirector
from petal.core.config.agent import AgentConfig, StepConfig
from petal.core.factory import Agent
from petal.core.steps.registry import StepRegistry
from typing_extensions import Annotated


class StateWithMessages(TypedDict):
    messages: Annotated[list, add_messages]
    name: str


def dummy_step_function(x):
    return x


def llm_config():
    return {"provider": "openai", "model": "gpt-3.5-turbo"}


def default_agent_config_kwargs():
    return dict(name=None, memory=None, llm_config=None)


def default_step_kwargs():
    return dict(node_name=None)


class TestAgentBuilderDirector:
    @pytest.fixture
    def sample_config(self):
        return AgentConfig(
            state_type=StateWithMessages,
            steps=[
                StepConfig(
                    strategy_type="llm",
                    config={"prompt_template": "Hello", "llm_config": llm_config()},
                    **default_step_kwargs(),
                ),
                StepConfig(
                    strategy_type="custom",
                    config={"step_function": dummy_step_function},
                    **default_step_kwargs(),
                ),
            ],
            **default_agent_config_kwargs(),
        )

    @pytest.fixture
    def sample_registry(self):
        registry = StepRegistry()
        return registry

    @pytest.fixture
    def director(self, sample_config, sample_registry):
        return AgentBuilderDirector(sample_config, sample_registry)

    def test_director_initialization(self, sample_config, sample_registry):
        director = AgentBuilderDirector(sample_config, sample_registry)
        assert director.config == sample_config
        assert director.registry == sample_registry

    def test_director_initialization_with_none_config(self, sample_registry):
        with pytest.raises(ValueError, match="config cannot be None"):
            AgentBuilderDirector(None, sample_registry)  # type: ignore[arg-type]

    def test_director_initialization_with_none_registry(self, sample_config):
        with pytest.raises(ValueError, match="registry cannot be None"):
            AgentBuilderDirector(sample_config, None)  # type: ignore[arg-type]

    def test_build_with_empty_steps_raises_error(self, sample_registry):
        config = AgentConfig(
            state_type=StateWithMessages, steps=[], **default_agent_config_kwargs()
        )
        director = AgentBuilderDirector(config, sample_registry)
        with pytest.raises(ValueError, match="Cannot build agent: no steps configured"):
            director.build()

    def test_create_state_type_with_llm_steps(self, director):
        state_type = director._create_state_type()
        assert hasattr(state_type, "__annotations__")
        assert "messages" in state_type.__annotations__

    def test_create_state_type_without_llm_steps(self, sample_registry):
        config = AgentConfig(
            state_type=StateWithMessages,
            steps=[
                StepConfig(
                    strategy_type="custom",
                    config={"step_function": dummy_step_function},
                    **default_step_kwargs(),
                )
            ],
            **default_agent_config_kwargs(),
        )
        director = AgentBuilderDirector(config, sample_registry)
        state_type = director._create_state_type()
        assert state_type == StateWithMessages

    def test_create_state_type_with_existing_messages(self, sample_registry):
        class StateWithMessages2(TypedDict):
            messages: Annotated[list, add_messages]
            name: str

        config = AgentConfig(
            state_type=StateWithMessages2,
            steps=[
                StepConfig(
                    strategy_type="llm",
                    config={"prompt_template": "Hello", "llm_config": llm_config()},
                    **default_step_kwargs(),
                )
            ],
            **default_agent_config_kwargs(),
        )
        director = AgentBuilderDirector(config, sample_registry)
        state_type = director._create_state_type()
        assert state_type == StateWithMessages2

    def test_build_graph_creates_correct_structure(self, director):
        """Test that _build_graph creates a real StateGraph with correct structure."""
        # Use real StateGraph instead of mocking
        graph = director._build_graph(StateWithMessages)

        # Verify we got a compiled graph (Runnable)
        assert hasattr(graph, "invoke")
        assert callable(graph.invoke)
        assert hasattr(graph, "ainvoke")
        assert callable(graph.ainvoke)

        # The graph should be a CompiledStateGraph, not directly callable
        assert hasattr(graph, "__class__")
        assert "CompiledStateGraph" in str(graph.__class__)

    def test_build_graph_with_custom_node_names(self, sample_registry):
        """Test that _build_graph respects custom node names."""
        config = AgentConfig(
            state_type=StateWithMessages,
            steps=[
                StepConfig(
                    strategy_type="llm",
                    config={"prompt_template": "Hello", "llm_config": llm_config()},
                    node_name="custom_llm_node",
                ),
                StepConfig(
                    strategy_type="custom",
                    config={"step_function": dummy_step_function},
                    node_name="custom_function_node",
                ),
            ],
            **default_agent_config_kwargs(),
        )
        director = AgentBuilderDirector(config, sample_registry)

        # Build real graph
        graph = director._build_graph(StateWithMessages)

        # Verify we got a compiled graph
        assert hasattr(graph, "invoke")
        assert callable(graph.invoke)
        assert hasattr(graph, "ainvoke")
        assert callable(graph.ainvoke)

        # The graph should be a CompiledStateGraph
        assert hasattr(graph, "__class__")
        assert "CompiledStateGraph" in str(graph.__class__)

    def test_build_graph_compiles_graph(self, director):
        """Test that _build_graph returns a compiled graph."""
        graph = director._build_graph(StateWithMessages)

        # A compiled graph should have invoke and ainvoke methods
        assert callable(graph.invoke)
        assert callable(graph.ainvoke)

        # Should be a CompiledStateGraph
        assert hasattr(graph, "__class__")
        assert "CompiledStateGraph" in str(graph.__class__)

    def test_build_creates_agent_with_real_integration(self, director):
        """Test the full build process without mocking internal methods."""
        # This test exercises the real build process
        agent = director.build()

        # Verify we got a real Agent
        assert isinstance(agent, Agent)
        assert agent.built is True

        # The agent should have a graph that's a CompiledStateGraph
        assert hasattr(agent, "graph")
        assert agent.graph is not None
        assert hasattr(agent.graph, "ainvoke")
        assert callable(agent.graph.ainvoke)

    def test_build_validates_configuration(self, sample_registry):
        config = AgentConfig(
            state_type=StateWithMessages, steps=[], **default_agent_config_kwargs()
        )
        director = AgentBuilderDirector(config, sample_registry)
        with pytest.raises(ValueError, match="Cannot build agent: no steps configured"):
            director.build()

    def test_build_with_invalid_step_type_raises_error(
        self, sample_config, sample_registry
    ):
        sample_config.steps.append(
            StepConfig(strategy_type="invalid_type", config={}, **default_step_kwargs())
        )
        director = AgentBuilderDirector(sample_config, sample_registry)
        with pytest.raises(ValueError, match="Unknown step type: invalid_type"):
            director.build()

    def test_validate_configuration_with_valid_config(self, director):
        director._validate_configuration()

    def test_validate_configuration_with_empty_steps(self, sample_registry):
        config = AgentConfig(
            state_type=StateWithMessages, steps=[], **default_agent_config_kwargs()
        )
        director = AgentBuilderDirector(config, sample_registry)
        with pytest.raises(ValueError, match="Cannot build agent: no steps configured"):
            director._validate_configuration()

    def test_validate_configuration_with_invalid_step_type(
        self, sample_config, sample_registry
    ):
        sample_config.steps.append(
            StepConfig(strategy_type="invalid_type", config={}, **default_step_kwargs())
        )
        director = AgentBuilderDirector(sample_config, sample_registry)
        with pytest.raises(
            ValueError, match="Invalid step 2: Unknown step type: invalid_type"
        ):
            director._validate_configuration()

    def test_build_integration_with_real_strategies(self, sample_registry):
        config = AgentConfig(
            state_type=StateWithMessages,
            steps=[
                StepConfig(
                    strategy_type="llm",
                    config={"prompt_template": "Hello", "llm_config": llm_config()},
                    **default_step_kwargs(),
                ),
                StepConfig(
                    strategy_type="custom",
                    config={"step_function": dummy_step_function},
                    **default_step_kwargs(),
                ),
            ],
            **default_agent_config_kwargs(),
        )
        director = AgentBuilderDirector(config, sample_registry)
        agent = director.build()
        assert isinstance(agent, Agent)
        assert agent.built is True

    def test_build_with_single_step(self, sample_registry):
        config = AgentConfig(
            state_type=StateWithMessages,
            steps=[
                StepConfig(
                    strategy_type="custom",
                    config={"step_function": dummy_step_function},
                    **default_step_kwargs(),
                )
            ],
            **default_agent_config_kwargs(),
        )
        director = AgentBuilderDirector(config, sample_registry)
        agent = director.build()
        assert isinstance(agent, Agent)
        assert agent.built is True

    def test_build_with_multiple_steps(self, sample_registry):
        config = AgentConfig(
            state_type=StateWithMessages,
            steps=[
                StepConfig(
                    strategy_type="llm",
                    config={"prompt_template": "Step 1", "llm_config": llm_config()},
                    **default_step_kwargs(),
                ),
                StepConfig(
                    strategy_type="custom",
                    config={"step_function": dummy_step_function},
                    **default_step_kwargs(),
                ),
                StepConfig(
                    strategy_type="llm",
                    config={"prompt_template": "Step 3", "llm_config": llm_config()},
                    **default_step_kwargs(),
                ),
            ],
            **default_agent_config_kwargs(),
        )
        director = AgentBuilderDirector(config, sample_registry)
        agent = director.build()
        assert isinstance(agent, Agent)
        assert agent.built is True

    def test_build_graph_with_real_state_type_creation(self, sample_registry):
        """Test that _build_graph works with dynamically created state types."""

        # Create a base state type without messages
        class BaseState(TypedDict):
            name: str
            value: int

        config = AgentConfig(
            state_type=BaseState,
            steps=[
                StepConfig(
                    strategy_type="llm",
                    config={"prompt_template": "Hello", "llm_config": llm_config()},
                    **default_step_kwargs(),
                ),
            ],
            **default_agent_config_kwargs(),
        )
        director = AgentBuilderDirector(config, sample_registry)

        # This should create a state type with messages added
        state_type = director._create_state_type()
        assert "messages" in state_type.__annotations__
        assert "name" in state_type.__annotations__
        assert "value" in state_type.__annotations__

        # Build graph with the enhanced state type
        graph = director._build_graph(state_type)
        assert hasattr(graph, "ainvoke")
        assert callable(graph.ainvoke)
        assert hasattr(graph, "__class__")
        assert "CompiledStateGraph" in str(graph.__class__)
