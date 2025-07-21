"""Tests for checkpointer integration with agent building."""

from unittest.mock import MagicMock, patch

import pytest
from petal.core.builders.director import AgentBuilderDirector
from petal.core.config.agent import AgentConfig, StepConfig
from petal.core.config.checkpointer import CheckpointerConfig
from petal.core.steps.registry import StepRegistry


class TestCheckpointerIntegration:
    """Test checkpointer integration with agent building."""

    @pytest.fixture
    def basic_config(self) -> AgentConfig:
        """Create a basic agent config with a custom step."""
        return AgentConfig(
            name="test_agent",
            state_type=dict,
            steps=[],
            memory=None,
            llm_config=None,
            checkpointer=CheckpointerConfig(type="memory"),
        )

    @pytest.fixture
    def registry(self) -> StepRegistry:
        """Create a step registry."""
        return StepRegistry()

    @pytest.mark.asyncio
    async def test_director_with_memory_checkpointer(self, basic_config, registry):
        """Test director builds agent with memory checkpointer."""
        # Add a simple custom step
        step_config = StepConfig(
            strategy_type="custom",
            config={"step_function": lambda x: x},
            node_name=None,
        )
        basic_config.add_step(step_config)

        director = AgentBuilderDirector(basic_config, registry)
        agent = await director.build()

        # Verify agent was built successfully
        assert agent is not None
        assert hasattr(agent, "graph")

    @pytest.mark.asyncio
    async def test_director_without_checkpointer(self, registry):
        """Test director builds agent without checkpointer (default behavior)."""
        config = AgentConfig(
            name="test_agent", state_type=dict, steps=[], memory=None, llm_config=None
        )

        # Add a simple custom step
        step_config = StepConfig(
            strategy_type="custom",
            config={"step_function": lambda x: x},
            node_name=None,
        )
        config.add_step(step_config)

        director = AgentBuilderDirector(config, registry)
        agent = await director.build()

        # Verify agent was built successfully
        assert agent is not None
        assert hasattr(agent, "graph")

    @pytest.mark.asyncio
    async def test_director_with_disabled_checkpointer(self, registry):
        """Test director builds agent with disabled checkpointer."""
        config = AgentConfig(
            name="test_agent",
            state_type=dict,
            steps=[],
            memory=None,
            llm_config=None,
            checkpointer=CheckpointerConfig(type="memory", enabled=False),
        )

        # Add a simple custom step
        step_config = StepConfig(
            strategy_type="custom",
            config={"step_function": lambda x: x},
            node_name=None,
        )
        config.add_step(step_config)

        director = AgentBuilderDirector(config, registry)
        agent = await director.build()

        # Verify agent was built successfully
        assert agent is not None
        assert hasattr(agent, "graph")

    @pytest.mark.asyncio
    async def test_checkpointer_is_passed_to_graph_compile(self, registry):
        """Test that checkpointer is actually passed to graph.compile()."""
        config = AgentConfig(
            name="test_agent",
            state_type=dict,
            steps=[],
            memory=None,
            llm_config=None,
            checkpointer=CheckpointerConfig(type="memory"),
        )

        # Add a simple custom step
        step_config = StepConfig(
            strategy_type="custom",
            config={"step_function": lambda x: x},
            node_name=None,
        )
        config.add_step(step_config)

        director = AgentBuilderDirector(config, registry)

        # Mock the graph.compile method to verify checkpointer is passed
        with patch("langgraph.graph.StateGraph.compile") as mock_compile:
            mock_compile.return_value = MagicMock()
            await director.build()

            # Verify compile was called with checkpointer
            mock_compile.assert_called_once()
            call_args = mock_compile.call_args
            assert "checkpointer" in call_args.kwargs
            assert call_args.kwargs["checkpointer"] is not None

    @pytest.mark.asyncio
    async def test_checkpointer_not_passed_when_disabled(self, registry):
        """Test that checkpointer is not passed when disabled."""
        config = AgentConfig(
            name="test_agent",
            state_type=dict,
            steps=[],
            memory=None,
            llm_config=None,
            checkpointer=CheckpointerConfig(type="memory", enabled=False),
        )

        # Add a simple custom step
        step_config = StepConfig(
            strategy_type="custom",
            config={"step_function": lambda x: x},
            node_name=None,
        )
        config.add_step(step_config)

        director = AgentBuilderDirector(config, registry)

        # Mock the graph.compile method to verify checkpointer is not passed
        with patch("langgraph.graph.StateGraph.compile") as mock_compile:
            mock_compile.return_value = MagicMock()
            await director.build()

            # Verify compile was called without checkpointer
            mock_compile.assert_called_once()
            call_args = mock_compile.call_args
            assert "checkpointer" not in call_args.kwargs
