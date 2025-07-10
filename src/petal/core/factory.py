from typing import Any, Callable, Dict, Optional

from langchain.chat_models.base import BaseChatModel
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from petal.core.agent import Agent
from petal.core.builders.agent import AgentBuilder
from petal.core.config.agent import LLMConfig


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

    def add(
        self, step: Callable[..., Any], node_name: Optional[str] = None
    ) -> "AgentFactory":
        """
        Add an async step to the agent. If no node_name is provided, one will be auto-generated.
        """
        self._builder.with_step("custom", step_function=step, node_name=node_name)
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
        Set the prompt template for the most recently added LLM step.
        """
        if not self._builder._config.steps:
            raise ValueError("No steps have been added to configure prompt for.")
        step_config = self._builder._config.steps[-1]
        if step_config.strategy_type != "llm":
            raise ValueError("The most recent step is not an LLM step.")
        step_config.config["prompt_template"] = prompt_template
        return self

    def with_system_prompt(self, system_prompt: str) -> "AgentFactory":
        """
        Set the system prompt for the most recently added LLM step.
        """
        if not self._builder._config.steps:
            raise ValueError("No steps have been added to configure system prompt for.")
        step_config = self._builder._config.steps[-1]
        if step_config.strategy_type != "llm":
            raise ValueError("The most recent step is not an LLM step.")
        step_config.config["system_prompt"] = system_prompt
        return self

    def with_structured_output(
        self, model: Any, key: Optional[str] = None
    ) -> "AgentFactory":
        """
        Bind a structured output schema (Pydantic model) to the most recent LLM step.
        Optionally wrap the output in a dict with the given key.
        """
        self._builder.with_structured_output(model, key)
        return self

    def build(self) -> Agent:
        """Build the agent using new architecture."""
        return self._builder.build()
