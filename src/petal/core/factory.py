from typing import Annotated, Any, Callable, Dict, List, Optional, Union

from langchain.chat_models.base import BaseChatModel
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from petal.core.agent import Agent
from petal.core.builders.agent import AgentBuilder
from petal.core.config.agent import LLMConfig
from petal.core.config.checkpointer import CheckpointerConfig
from petal.core.registry import ToolRegistry


class ToolDiscoveryConfig(TypedDict):
    enabled: bool
    folders: Optional[List[str]]
    config_locations: Optional[List[str]]
    exclude_patterns: Optional[List[str]]


class DefaultState(TypedDict):
    """Default state schema for agents."""

    messages: Annotated[list, add_messages]
    name: str


class AgentFactory:
    """
    Builder and fluent interface for constructing Agent objects as LangGraph StateGraphs.
    """

    def __init__(self, state_type: type):
        if state_type is None:
            raise TypeError("state_type is required and cannot be None")

        # Use new architecture internally
        self._builder = AgentBuilder(state_type)

        # Initialize tool registry (singleton)
        self._tool_registry = ToolRegistry()

        # Initialize tool discovery configuration with defaults
        self._tool_discovery_config: ToolDiscoveryConfig = {
            "enabled": True,
            "folders": None,
            "config_locations": None,
            "exclude_patterns": None,
        }

    def add(
        self, step: Callable[..., Any], node_name: Optional[str] = None
    ) -> "AgentFactory":
        """
        Add an async step to the agent. If no node_name is provided, one will be auto-generated.
        """
        self._builder.with_step("custom", step_function=step, node_name=node_name)
        return self

    def with_checkpointer(
        self, checkpointer_config: CheckpointerConfig
    ) -> "AgentFactory":
        """
        Configure checkpointer for state persistence.

        Args:
            checkpointer_config: The checkpointer configuration

        Returns:
            self: For method chaining
        """
        self._builder._config.set_checkpointer(checkpointer_config)
        return self

    def with_tools(
        self, tools: List[Union[str, Any]], scratchpad_key: Optional[str] = None
    ) -> "AgentFactory":
        """
        Add tools to the most recent LLM step with optional scratchpad support.

        Args:
            tools: List of tool names (strings) or tool objects. String names are resolved via ToolFactory.
            scratchpad_key: Optional key for storing tool observations in state

        Raises:
            ValueError: If no steps have been added, the most recent step is not an LLM step, or tools list is empty
        """
        if not self._builder._config.steps:
            raise ValueError("No steps have been added. Call with_chat() first.")

        last_step = self._builder._config.steps[-1]
        if last_step.strategy_type != "llm":
            raise ValueError(
                "The most recent step is not an LLM step. Call with_chat() first."
            )

        if not tools:
            raise ValueError("Tools list cannot be empty. Provide at least one tool.")

        # Store tools as-is (strings or objects) - resolve at build time
        last_step.config["tools"] = tools
        if scratchpad_key:
            last_step.config["scratchpad_key"] = scratchpad_key

        return self

    def with_react_tools(
        self, tools: List[Union[str, Any]], scratchpad_key: str = "scratchpad"
    ) -> "AgentFactory":
        """
        Add tools with ReAct-style scratchpad for persistent tool observation history.

        Args:
            tools: List of tool names (strings) or tool objects. String names are resolved via ToolFactory.
            scratchpad_key: Key for storing tool observations in state (defaults to "scratchpad")
        """
        return self.with_tools(tools, scratchpad_key=scratchpad_key)

    def with_react_loop(self, tools: List[Union[str, Any]], **config) -> "AgentFactory":
        """
        Add a React reasoning loop step.

        Args:
            tools: List of tool names (strings) or tool objects. String names are resolved via ToolFactory.
            **config: Additional configuration for the React step
        """
        # Add state_schema to config for React step
        config["state_schema"] = self._builder._config.state_type
        self._builder.with_step("react", tools=tools, **config)
        return self

    def with_chat(
        self,
        llm: Optional[Any] = None,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        llm_config: Optional[Dict[str, Any] | LLMConfig] = None,
        **kwargs,
    ) -> "AgentFactory":
        """
        Adds an LLM step to the chain. Accepts prompt_template, system_prompt, and llm_config (as a dict or LLMConfig) as direct arguments for step configuration. Returns self for fluent chaining.
        """
        config: Dict[str, Any] = {}
        if llm is not None:
            if not isinstance(llm, BaseChatModel):
                raise ValueError(f"llm must be a BaseChatModel, not {type(llm)}")
            config["llm_instance"] = llm  # type: ignore[assignment]
        if llm_config is not None:
            if isinstance(llm_config, LLMConfig):
                config.update(llm_config.model_dump())
            else:
                config.update(llm_config)
        if prompt_template is not None:
            config["prompt_template"] = prompt_template  # type: ignore[assignment]
        if system_prompt is not None:
            config["system_prompt"] = system_prompt  # type: ignore[assignment]
        config.update(kwargs)
        self._builder.with_step("llm", node_name=None, **config)
        return self

    def with_prompt(self, prompt_template: str) -> "AgentFactory":
        """
        Set the prompt template for the most recently added LLM or React step.
        """
        if not self._builder._config.steps:
            raise ValueError("No steps have been added to configure prompt for.")
        step_config = self._builder._config.steps[-1]
        if step_config.strategy_type not in ["llm", "react"]:
            raise ValueError("The most recent step is not an LLM or React step.")
        step_config.config["prompt_template"] = prompt_template
        return self

    def with_system_prompt(self, system_prompt: str) -> "AgentFactory":
        """
        Set the system prompt for the most recently added LLM or React step.
        """
        if not self._builder._config.steps:
            raise ValueError("No steps have been added to configure system prompt for.")
        step_config = self._builder._config.steps[-1]
        if step_config.strategy_type not in ["llm", "react"]:
            raise ValueError("The most recent step is not an LLM or React step.")
        step_config.config["system_prompt"] = system_prompt
        return self

    def with_structured_output(
        self, model: Any, key: Optional[str] = None
    ) -> "AgentFactory":
        """
        Bind a structured output schema (Pydantic model) to the most recent LLM or React step.
        Optionally wrap the output in a dict with the given key.
        """
        self._builder.with_structured_output(model, key)
        return self

    def with_tool_discovery(
        self,
        enabled: bool = True,
        folders: Optional[List[str]] = None,
        config_locations: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> "AgentFactory":
        """
        Configure tool discovery for this agent factory.

        Args:
            enabled: Whether to enable tool discovery
            folders: List of folders to scan for tools
            config_locations: List of config file locations to scan
            exclude_patterns: List of patterns to exclude from discovery

        Returns:
            self for fluent chaining
        """
        self._tool_discovery_config.update(
            {
                "enabled": enabled,
                "folders": folders,
                "config_locations": config_locations,
                "exclude_patterns": exclude_patterns,
            }
        )
        return self

    async def build(self) -> Agent:
        """Build the agent using new architecture."""
        # Add tool steps for each LLM step that has tools
        for step_config in self._builder._config.steps:
            if step_config.strategy_type == "llm" and "tools" in step_config.config:
                # Create tool step for this LLM step
                tool_config = {
                    "tools": step_config.config["tools"],
                    "scratchpad_key": step_config.config.get("scratchpad_key"),
                }
                self._builder.with_step("tool", **tool_config)

        return await self._builder.build()

    @staticmethod
    def diagram_agent(agent: "Agent", path: str, format: str = "png") -> None:
        """
        Generate a diagram of an agent's graph and save it to a file.

        Args:
            agent: The built Agent instance
            path: File path where the diagram should be saved
            format: Output format (png, svg, pdf, etc.)

        Raises:
            RuntimeError: If diagram generation fails
        """
        if not agent.built or agent.graph is None:
            raise RuntimeError("Agent must be built before generating diagram")

        try:
            # The compiled graph has a get_graph() method that returns a Graph object
            graph_obj = agent.graph.get_graph()
            if hasattr(graph_obj, "draw_mermaid_png"):
                if format.lower() == "png":
                    img_bytes = graph_obj.draw_mermaid_png()
                elif format.lower() == "svg":
                    img_bytes = graph_obj.draw_mermaid_svg()
                else:
                    raise RuntimeError(f"Unsupported format: {format}")

                with open(path, "wb") as f:
                    f.write(img_bytes)
            else:
                raise RuntimeError(
                    "Graph object doesn't support mermaid diagram generation"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to generate diagram: {e}") from e

    async def diagram_graph(self, path: str, format: str = "png") -> Any:
        """
        Generate a diagram of the agent graph and save it to a file.

        This method builds the agent and generates a visualization using LangGraph's
        built-in diagram capabilities.

        Args:
            path: File path where the diagram should be saved
            format: Output format (png, svg, pdf, etc.)

        Raises:
            ValueError: If no steps have been configured
            RuntimeError: If diagram generation fails
        """
        if not self._builder._config.steps:
            raise ValueError("Cannot generate diagram: no steps have been configured")

        # Build the agent
        agent = await self.build()

        # Generate diagram using the static method
        AgentFactory.diagram_agent(agent, path, format)

    async def node_from_yaml(self, path: str):
        """
        Load a node from YAML configuration file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Callable node function

        Raises:
            YAMLFileNotFoundError: If YAML file doesn't exist
            YAMLParseError: If YAML is invalid
            ValueError: If node type is unsupported
        """
        from petal.core.yaml.handlers import HandlerFactory
        from petal.core.yaml.parser import YAMLNodeParser

        # Parse YAML configuration
        parser = YAMLNodeParser()
        config = parser.parse_node_config(path)

        # Create appropriate handler
        handler_factory = HandlerFactory()
        handler = handler_factory.get_handler(config.type)

        # Create the node function
        node_function = await handler.create_node(config)

        # Add the node to the builder (following the same pattern as other methods)
        self._builder.with_step(
            "custom", step_function=node_function, node_name=config.name
        )

        return node_function
