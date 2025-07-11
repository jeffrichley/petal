"""Director for building agents from configuration."""

import logging
from typing import Any, Type

from langgraph.graph import END, StateGraph

from petal.core.agent import Agent
from petal.core.config.agent import AgentConfig
from petal.core.config.state import StateTypeFactory
from petal.core.steps.registry import StepRegistry

logger = logging.getLogger(__name__)


class AgentBuilderDirector:
    """
    Director class that orchestrates the building of Agent objects from configurations.
    Uses the Builder pattern to construct complex agent graphs.
    """

    def __init__(self, config: AgentConfig, registry: StepRegistry):
        self.config = config
        self.registry = registry

    def build(self) -> Agent:
        """
        Build an Agent from the configuration.

        Returns:
            Agent: The built agent instance

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_configuration()

        # Create state type from configuration
        state_type = self._create_state_type()

        # Build the graph
        graph = self._build_graph(state_type)

        # Create and return the agent
        agent = Agent(
            graph=graph,
            config=self.config,
            built=True,
        )

        return agent

    def _build_graph(self, state_type: Type) -> Any:
        """
        Build the LangGraph StateGraph from the configuration.

        Args:
            state_type: The state type for the graph

        Returns:
            The compiled graph
        """
        # Build the uncompiled graph
        graph = self._build_uncompiled_graph(state_type)

        # Compile and return
        return graph.compile()

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

    def _build_uncompiled_graph(self, state_type: Type) -> StateGraph:
        """
        Build the uncompiled StateGraph.

        Args:
            state_type: The state type for the graph

        Returns:
            The uncompiled StateGraph
        """
        graph = StateGraph(state_type)

        # Track nodes and edges for visualization
        node_names = []
        step_types = []
        hard_edges = []
        conditional_edges = []
        react_pairs = []

        # Add nodes for each step
        for i, step_config in enumerate(self.config.steps):
            node_name = step_config.node_name or f"step_{i}"
            node_names.append(node_name)
            step_types.append(step_config.strategy_type)

            # Create the step using the registry
            step = self.registry.create_step(step_config)
            graph.add_node(node_name, step)

            # Handle edges based on step type
            if step_config.strategy_type == "llm":
                self._handle_llm_step(
                    graph,
                    node_name,
                    i,
                    node_names,
                    step_types,
                    react_pairs,
                    hard_edges,
                    conditional_edges,
                )
            elif step_config.strategy_type == "tool":
                self._handle_tool_step(
                    graph, node_name, i, node_names, step_types, react_pairs, hard_edges
                )
            elif step_config.strategy_type == "custom":
                self._handle_custom_step(
                    graph, node_name, i, node_names, step_types, hard_edges
                )

        # Set entry point
        if node_names:
            graph.set_entry_point(node_names[0])

        # Print graph structure for debugging
        self._print_graph_structure(
            node_names, step_types, hard_edges, conditional_edges
        )

        return graph

    def _handle_llm_step(
        self,
        graph,
        node_name,
        index,
        node_names,
        step_types,
        react_pairs,
        hard_edges,
        conditional_edges,
    ):
        """Handle LLM step edge creation."""
        # Check if this LLM has a tool step after it
        is_in_react_loop = any(node_name == llm for llm, _ in react_pairs)

        if is_in_react_loop:
            # This LLM is part of a ReAct loop
            tool_node = next(tool for llm, tool in react_pairs if llm == node_name)

            # Create conditional edge function
            def create_tool_condition(tool_node_name, next_step_name):
                def tool_condition(state):
                    from petal.core.steps.tool import decide_next_step

                    return decide_next_step(state, tool_node_name, next_step_name)

                return tool_condition

            # Add conditional edge: LLM -> Tool OR next step OR END
            next_step = (
                self._get_next_non_react_step(
                    index, node_names, step_types, react_pairs
                )
                or END
            )

            # LLM can go to tool OR next step (which could be END)
            graph.add_conditional_edges(
                node_name, create_tool_condition(tool_node, next_step)
            )
            conditional_edges.append((node_name, tool_node, "tool_condition"))
            conditional_edges.append((node_name, next_step, "tool_condition"))
        else:
            # This LLM is not part of a ReAct loop
            if index + 1 < len(node_names):
                # Connect to next step
                next_node = node_names[index + 1]
                graph.add_edge(node_name, next_node)
                hard_edges.append((node_name, next_node))
            else:
                # Connect to END
                graph.add_edge(node_name, END)
                hard_edges.append((node_name, END))

    def _handle_tool_step(
        self, graph, node_name, index, node_names, step_types, react_pairs, hard_edges
    ):
        """Handle Tool step edge creation."""
        # Find the LLM that calls this tool
        calling_llm = next(llm for llm, tool in react_pairs if tool == node_name)

        # Tool always goes back to its calling LLM
        graph.add_edge(node_name, calling_llm)
        hard_edges.append((node_name, calling_llm))

        # Log the tool step for debugging
        logger.debug(
            f"Tool step {node_name} (index {index}) connected to LLM {calling_llm}"
        )
        logger.debug(f"Available nodes: {node_names}, step types: {step_types}")

    def _handle_custom_step(
        self, graph, node_name, index, node_names, step_types, hard_edges
    ):
        """Handle Custom step edge creation."""
        if index + 1 < len(node_names):
            # Connect to next step
            next_node = node_names[index + 1]
            graph.add_edge(node_name, next_node)
            hard_edges.append((node_name, next_node))
        else:
            # Connect to END
            graph.add_edge(node_name, END)
            hard_edges.append((node_name, END))

        # Log the custom step for debugging
        logger.debug(
            f"Custom step {node_name} (index {index}) of type {step_types[index] if index < len(step_types) else 'unknown'}"
        )

    def _get_next_non_react_step(
        self, current_index, node_names, step_types, react_pairs
    ):
        """Get the next step that's not part of the current ReAct loop."""
        # Log react pairs for debugging
        if react_pairs:
            logger.debug(f"ReAct pairs: {react_pairs}")

        for i in range(current_index + 1, len(node_names)):
            node_name = node_names[i]
            step_type = step_types[i]

            # Skip tool steps (they're part of ReAct loops)
            if step_type == "tool":
                continue

            # Return the first non-tool step
            return node_name

        return None

    def _print_graph_structure(
        self,
        node_names: list,
        step_types: list,
        hard_edges: list,
        conditional_edges: list,
    ) -> None:
        """
        Print a rich visualization of the graph structure.

        Args:
            node_names: List of node names
            step_types: List of step types
            hard_edges: List of hard edges (from, to)
            conditional_edges: List of conditional edges (from, to, condition)
        """
        # ANSI color codes - using lowercase to follow Python conventions
        reset = "\033[0m"
        bold = "\033[1m"
        blue = "\033[94m"
        green = "\033[92m"
        yellow = "\033[93m"
        red = "\033[91m"
        cyan = "\033[96m"
        magenta = "\033[95m"
        white = "\033[97m"

        print(f"\n{bold}{blue}{'='*60}{reset}")
        print(f"{bold}{cyan}🔗 AGENT GRAPH STRUCTURE{reset}")
        print(f"{bold}{blue}{'='*60}{reset}")

        # Print nodes
        print(f"\n{bold}{green}📋 NODES:{reset}")
        print(f"{green}{'-' * 40}{reset}")
        for i, (node_name, step_type) in enumerate(
            zip(node_names, step_types, strict=False)
        ):
            step_color = yellow if step_type == "llm" else magenta
            print(
                f"  {cyan}{i+1:2d}.{reset} {white}{node_name:<20}{reset} ({step_color}{step_type}{reset})"
            )

        # Print hard edges
        print(f"\n{bold}{green}🔗 HARD EDGES (always taken):{reset}")
        print(f"{green}{'-' * 40}{reset}")
        for from_node, to_node in hard_edges:
            from_display = "START" if from_node == "START" else from_node
            to_display = "END" if to_node == "END" else to_node
            from_color = yellow if from_display == "START" else white
            to_color = red if to_display == "END" else white
            print(
                f"  {from_color}{from_display:<20}{reset} {cyan}→{reset} {to_color}{to_display}{reset}"
            )

        # Print conditional edges
        if conditional_edges:
            print(f"\n{bold}{green}🔄 CONDITIONAL EDGES (conditionally taken):{reset}")
            print(f"{green}{'-' * 40}{reset}")
            for from_node, to_node, condition in conditional_edges:
                print(
                    f"  {white}{from_node:<20}{reset} {cyan}→{reset} {white}{to_node:<20}{reset} {yellow}[{condition}]{reset}"
                )
        else:
            print(f"\n{bold}{green}🔄 CONDITIONAL EDGES:{reset} {yellow}None{reset}")

        # Print graph summary
        print(f"\n{bold}{green}📊 GRAPH SUMMARY:{reset}")
        print(f"{green}{'-' * 40}{reset}")
        print(f"  {cyan}Total Nodes:{reset}     {white}{len(node_names)}{reset}")
        print(f"  {cyan}Hard Edges:{reset}      {white}{len(hard_edges)}{reset}")
        print(
            f"  {cyan}Conditional Edges:{reset} {white}{len(conditional_edges)}{reset}"
        )

        # Show ReAct loops if any
        react_loops = [edge for edge in conditional_edges if "tool" in edge[2]]
        if react_loops:
            print(f"  {cyan}ReAct Loops:{reset}     {white}{len(react_loops)}{reset}")
            for from_node, to_node, _ in react_loops:
                print(
                    f"    {magenta}{from_node}{reset} {cyan}↔{reset} {magenta}{to_node}{reset}"
                )

        print(f"{bold}{blue}{'='*60}{reset}\n")

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
