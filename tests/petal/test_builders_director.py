"""Tests for AgentBuilderDirector."""

from typing import TypedDict
from unittest.mock import Mock, patch

import pytest
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from petal.core.builders.director import AgentBuilderDirector
from petal.core.config.agent import AgentConfig, StepConfig
from petal.core.factory import Agent
from petal.core.steps.registry import StepRegistry


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
        with patch("petal.core.builders.director.StateGraph") as mock_state_graph:
            mock_graph = Mock()
            mock_state_graph.return_value = mock_graph
            director._build_graph(StateWithMessages)
            mock_state_graph.assert_called_once_with(StateWithMessages)
            assert mock_graph.add_node.call_count == 2
            assert mock_graph.add_edge.call_count == 3

    def test_build_graph_with_custom_node_names(self, sample_registry):
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
        with patch("petal.core.builders.director.StateGraph") as mock_state_graph:
            mock_graph = Mock()
            mock_state_graph.return_value = mock_graph
            director._build_graph(StateWithMessages)
            add_node_calls = mock_graph.add_node.call_args_list
            assert add_node_calls[0][0][0] == "custom_llm_node"
            assert add_node_calls[1][0][0] == "custom_function_node"

    def test_build_graph_compiles_graph(self, director):
        with patch("petal.core.builders.director.StateGraph") as mock_state_graph:
            mock_graph = Mock()
            mock_state_graph.return_value = mock_graph
            director._build_graph(StateWithMessages)
            mock_graph.compile.assert_called_once()

    def test_build_creates_agent(self, director):
        with (
            patch.object(director, "_create_state_type") as mock_create_state,
            patch.object(director, "_build_graph") as mock_build_graph,
        ):
            mock_create_state.return_value = StateWithMessages
            mock_graph = Mock()
            mock_build_graph.return_value = mock_graph
            agent = director.build()
            assert isinstance(agent, Agent)
            mock_create_state.assert_called_once()
            mock_build_graph.assert_called_once_with(StateWithMessages)

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
