"""Tool step strategy for agent workflows."""

from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolNode

from petal.core.registry import ToolRegistry
from petal.core.steps.base import StepStrategy


class ToolStep:
    """
    Encapsulates the configuration and logic for a tool execution step.
    """

    def __init__(self, tools: List[Any], scratchpad_key: Optional[str] = None):
        """
        Initialize the tool step.

        Args:
            tools: List of tools to execute
            scratchpad_key: Optional key for storing tool observations in state
        """
        self.tools = tools
        self.scratchpad_key = scratchpad_key
        self._tool_node = ToolNode(tools)

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tools and return updated state.

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        # Execute tools using LangGraph's ToolNode
        result = await self._tool_node.ainvoke(state)

        # Extract tool messages and add to scratchpad if configured
        tool_msgs = result.get("messages", [])
        obs_lines = []

        for msg in tool_msgs:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                obs_lines.append(f"Observation: {msg.content}")

        # Update scratchpad if configured
        if self.scratchpad_key:
            current_scratchpad = state.get(self.scratchpad_key, "")
            if obs_lines:
                new_scratchpad = (
                    current_scratchpad
                    + (
                        "\n" + "\n".join(obs_lines)
                        if current_scratchpad
                        else "\n".join(obs_lines)
                    )
                ).strip()
                result[self.scratchpad_key] = new_scratchpad
            else:
                # No new observations, preserve previous scratchpad
                result[self.scratchpad_key] = current_scratchpad

        return result


class ToolStepStrategy(StepStrategy):
    """
    Strategy for creating tool execution steps.
    """

    async def create_step(self, config: Dict[str, Any]) -> Callable:
        """
        Create a ToolStep from configuration.

        Args:
            config: Configuration dictionary containing:
                - tools: List of tools to execute (can be strings or tool objects)
                - scratchpad_key: Optional key for scratchpad in state

        Returns:
            ToolStep callable

        Raises:
            ValueError: If no tools are provided
            KeyError: If a tool name cannot be resolved
        """
        tools = config.get("tools", [])
        scratchpad_key = config.get("scratchpad_key")

        if not tools:
            raise ValueError("Tool step requires at least one tool")

        # Resolve string tool names to actual tools using async registry
        resolved_tools = []
        registry = ToolRegistry()

        for tool in tools:
            if isinstance(tool, str):
                # Use async registry resolution for discovery support
                resolved_tool = await registry.resolve(tool)
                resolved_tools.append(resolved_tool)
            else:
                resolved_tools.append(tool)

        return ToolStep(resolved_tools, scratchpad_key)

    def get_node_name(self, index: int) -> str:
        """
        Generate node name for tool step.

        Args:
            index: Step index

        Returns:
            Node name
        """
        return f"tool_step_{index}"


def decide_next_step(
    state: Dict[str, Any], tool_node_name: str = "tools", next_step: str = END
) -> str:
    """
    Decide whether to call tools or go to next step.

    This function checks if the last message has tool calls and routes accordingly.
    Based on the pattern from playground3.py and LangGraph documentation.

    Args:
        state: Current agent state
        tool_node_name: Name of the tool node to route to
        next_step: Name of the next step to go to if not using tools (can be "END")

    Returns:
        Next node name or END
    """
    messages = state.get("messages", [])
    if not messages:
        return next_step

    last_message = messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return tool_node_name

    return next_step


def tools_condition(state: Dict[str, Any]) -> str:
    """
    LangGraph-compatible tool condition function.

    This is a simplified version that works with LangGraph's tools_condition pattern.

    Args:
        state: Current agent state

    Returns:
        Next node name or END
    """
    return decide_next_step(state)
