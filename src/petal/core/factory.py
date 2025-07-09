from typing import Any, Callable, Dict, Optional, Union

from langchain.chat_models.base import BaseChatModel
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from petal.core.agent import Agent
from petal.core.builders.agent import AgentBuilder


class DefaultState(TypedDict):
    """Default state schema for agents."""

    messages: Annotated[list, add_messages]
    name: str


class NonChatState(TypedDict, total=False):
    """State for non-chat agents that allows any fields."""

    pass


class MergeableState(TypedDict, total=False):
    """State schema that supports merging of any fields."""

    # This allows any keys to be added and merged
    # We'll use a simple dict approach for now


class AgentFactory:
    """
    Builder and fluent interface for constructing Agent objects as LangGraph StateGraphs.
    """

    def __init__(self, state_type: type):
        if state_type is None:
            raise TypeError("state_type is required and cannot be None")

        # Use new architecture internally
        self._builder = AgentBuilder(state_type)
        self._state_type = state_type
        self._built = False

    def add(
        self, step: Callable[..., Any], node_name: Optional[str] = None
    ) -> "AgentFactory":
        """
        Add an async step to the agent. If no node_name is provided, one will be auto-generated.
        """
        # Use new architecture internally
        self._builder.with_step("custom", step_function=step, node_name=node_name)
        return self

    def with_chat(
        self, llm: Optional[Union[BaseChatModel, Dict[str, Any]]] = None, **kwargs
    ) -> "ChatStepBuilder":
        """
        Adds an LLM step to the chain. Returns a ChatStepBuilder for configuring this specific LLM step.
        """
        # Convert to new architecture format
        config = {}
        if llm is not None:
            if isinstance(llm, BaseChatModel):
                config["llm_instance"] = llm
            elif isinstance(llm, dict):
                config.update(llm)
        config.update(kwargs)

        self._builder.with_step("llm", node_name=None, **config)
        return ChatStepBuilder(self, len(self._builder._config.steps) - 1)

    def build(self) -> Agent:
        """Build the agent using new architecture."""
        return self._builder.build()


class ChatStepBuilder:
    """
    Builder for configuring a specific LLM step in the chain.
    """

    def __init__(self, factory: AgentFactory, step_index: int):
        self.factory = factory
        self.step_index = step_index

    def with_prompt(self, prompt_template: str) -> "ChatStepBuilder":
        """Set the prompt template for this specific LLM step."""
        # Update the step configuration in the builder
        step_config = self.factory._builder._config.steps[self.step_index]
        step_config.config["prompt_template"] = prompt_template
        return self

    def with_system_prompt(self, system_prompt: str) -> "ChatStepBuilder":
        """Set the system prompt for this specific LLM step."""
        # Update the step configuration in the builder
        step_config = self.factory._builder._config.steps[self.step_index]
        step_config.config["system_prompt"] = system_prompt
        return self

    def add(self, step: Callable[[Dict], Dict]) -> "AgentFactory":
        """Add another step to the factory."""
        return self.factory.add(step)

    def with_chat(
        self, llm: Optional[Union[BaseChatModel, Dict[str, Any]]] = None, **kwargs
    ) -> "ChatStepBuilder":
        """Add another LLM step to the factory."""
        return self.factory.with_chat(llm, **kwargs)

    def build(self) -> Agent:
        """Build the agent."""
        return self.factory.build()
