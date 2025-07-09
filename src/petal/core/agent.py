"""The runnable agent object, composed of a LangGraph StateGraph."""

from typing import Any, Optional


class Agent:
    """
    The runnable agent object, composed of a LangGraph StateGraph.
    """

    def __init__(self):
        self.graph: Optional[Any] = None
        self.state_type: Optional[type] = None
        self.built = False

    def build(self, graph: Any, state_type: Optional[type] = None) -> "Agent":
        """
        Build the agent with the given graph and state type.

        Args:
            graph: The compiled StateGraph
            state_type: The state type for the agent

        Returns:
            self: For method chaining
        """
        self.graph = graph
        self.state_type = state_type
        self.built = True
        return self

    async def arun(self, state: dict) -> dict:
        if not self.built:
            raise RuntimeError("Agent.arun() called before build()")
        if self.graph is None:
            raise RuntimeError("Agent.graph is None - agent was not properly built")
        return await self.graph.ainvoke(state)
