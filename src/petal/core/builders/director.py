"""Director for building agents from configuration."""

from typing import Type

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from petal.core.agent import Agent
from petal.core.config.agent import AgentConfig
from petal.core.config.checkpointer import CheckpointerConfig
from petal.core.config.state import StateTypeFactory
from petal.core.steps.registry import StepRegistry


class AgentBuilderDirector:
    """Director for building agents from configuration."""

    def __init__(self, config: AgentConfig, registry: StepRegistry):
        """
        Initialize the AgentBuilderDirector.

        Args:
            config: The agent configuration
            registry: The step registry

        Raises:
            ValueError: If config or registry is None
        """
        if config is None:
            raise ValueError("config cannot be None")
        if registry is None:
            raise ValueError("registry cannot be None")

        self.config = config
        self.registry = registry

    async def build(self) -> Agent:
        """
        Build an agent from configuration.

        Returns:
            The built agent

        Raises:
            ValueError: If configuration is invalid or no steps are configured
        """
        # Validate configuration
        self._validate_configuration()

        # Create state type
        state_type = self._create_state_type()

        # Build graph
        graph = await self._build_graph(state_type)

        # Create and return agent
        return Agent().build(graph, state_type)

    def _create_state_type(self) -> Type:
        """
        Create the appropriate state type based on configuration.

        Returns:
            The state type with message support if needed
        """
        # Check if any steps are LLM steps (require message support)
        has_llm_steps = any(step.strategy_type == "llm" for step in self.config.steps)

        if has_llm_steps:
            return StateTypeFactory.create_with_messages(self.config.state_type)
        return self.config.state_type

    async def _build_graph(self, state_type: Type) -> Runnable:
        """
        Build the LangGraph StateGraph from configuration.

        Args:
            state_type: The state type for the graph

        Returns:
            The compiled StateGraph as a Runnable
        """
        graph = StateGraph(state_type)

        # Add nodes for each step
        for i, step_config in enumerate(self.config.steps):
            # Get strategy and create step
            strategy = self.registry.get_strategy(step_config.strategy_type)
            step = await strategy.create_step(step_config.config)

            # Generate node name
            node_name = step_config.node_name or strategy.get_node_name(i)

            # Add node to graph
            graph.add_node(node_name, step)

            # Add edges
            if i == 0:
                # First node connects from START
                graph.add_edge(START, node_name)
            else:
                # Connect to previous node
                prev_step = self.config.steps[i - 1]
                prev_strategy = self.registry.get_strategy(prev_step.strategy_type)
                prev_node_name = prev_step.node_name or prev_strategy.get_node_name(
                    i - 1
                )
                graph.add_edge(prev_node_name, node_name)

            # Last node connects to END
            if i == len(self.config.steps) - 1:
                graph.add_edge(node_name, END)

        # Handle checkpointer configuration
        if self.config.checkpointer and self.config.checkpointer.enabled:
            checkpointer = await self._create_checkpointer(self.config.checkpointer)
            return graph.compile(checkpointer=checkpointer)
        else:
            return graph.compile()

    async def _create_checkpointer(self, checkpointer_config: CheckpointerConfig):
        """
        Create a LangGraph checkpointer based on configuration.

        Args:
            checkpointer_config: The checkpointer configuration

        Returns:
            A LangGraph checkpointer instance

        Raises:
            ValueError: If checkpointer type is not supported
        """
        if checkpointer_config.type == "memory":
            from langgraph.checkpoint.memory import InMemorySaver

            return InMemorySaver()
        elif checkpointer_config.type == "postgres":
            from langgraph.checkpoint.postgres import PostgresSaver

            if not checkpointer_config.config:
                raise ValueError(
                    "Postgres checkpointer requires configuration (connection_string)"
                )
            return PostgresSaver(**checkpointer_config.config)
        elif checkpointer_config.type == "sqlite":
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            if not checkpointer_config.config:
                raise ValueError("SQLite checkpointer requires configuration (db_file)")

            # Get the database file path from config
            db_file = checkpointer_config.config.get(
                "db_file", "./data/conversations.db"
            )

            # Create async connection
            conn = await aiosqlite.connect(db_file)
            return AsyncSqliteSaver(conn)
        else:
            raise ValueError(
                f"Unsupported checkpointer type: {checkpointer_config.type}"
            )

    def _validate_configuration(self) -> None:
        """
        Validate configuration before building.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.steps:
            raise ValueError("Cannot build agent: no steps configured")

        # Validate all step configurations
        for i, step_config in enumerate(self.config.steps):
            try:
                self.registry.validate_strategy(step_config.strategy_type)
            except ValueError as e:
                raise ValueError(f"Invalid step {i}: {e}") from e
