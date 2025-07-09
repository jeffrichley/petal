from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints

from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState, add_messages
from typing_extensions import Annotated, TypedDict


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


class LLMStep:
    """
    Encapsulates the configuration and logic for an LLM step.
    """

    def __init__(self, prompt_template, system_prompt, llm_config, llm_instance):
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.llm_config = llm_config
        self.llm_instance = llm_instance

    async def __call__(self, state):
        llm = self._create_llm_instance()
        llm_messages, user_prompt = self._build_llm_messages(state)
        response = await llm.ainvoke(llm_messages)
        return self._format_llm_response(response, user_prompt)

    def _build_llm_messages(self, state):
        original_messages = state.get("messages", [])
        llm_messages = []
        if self.system_prompt:
            llm_messages.append({"role": "system", "content": self.system_prompt})
        llm_messages.extend(original_messages)
        user_prompt = None
        if self.prompt_template:
            try:
                user_prompt = self.prompt_template.format(**state)
            except KeyError as e:
                missing_key = str(e).strip("'")
                raise ValueError(
                    f"Prompt template '{self.prompt_template}' requires key '{missing_key}' "
                    f"but it's not available in the state. Available keys: {list(state.keys())}"
                ) from e
            llm_messages.append({"role": "user", "content": user_prompt})
        return llm_messages, user_prompt

    def _create_llm_instance(self):
        if self.llm_instance is not None:
            return self.llm_instance
        config = self.llm_config or {}
        provider = config.get("provider", "openai")
        openai_config = {k: v for k, v in config.items() if k != "provider"}
        if "model" not in openai_config:
            openai_config["model"] = "gpt-4o-mini"
        if "temperature" not in openai_config:
            openai_config["temperature"] = 0
        if provider == "openai":
            return ChatOpenAI(**openai_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _format_llm_response(self, response, user_prompt):
        # Return the complete state with messages added
        # This ensures the messages field is always present in the final state
        if user_prompt:
            return {"messages": [{"role": "user", "content": user_prompt}, response]}
        else:
            return {"messages": [response]}


class Agent:
    """
    The runnable agent object, composed of a LangGraph StateGraph.
    """

    def __init__(self):
        self.graph = None
        self.state_type = None
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
        self.state_type = state_type or DefaultState
        self.built = True
        return self

    async def arun(self, state: dict) -> dict:
        if not self.built:
            raise RuntimeError("Agent.arun() called before build()")
        return await self.graph.ainvoke(state)


class AgentFactory:
    """
    Builder and fluent interface for constructing Agent objects as LangGraph StateGraphs.
    """

    _dynamic_state_type_cache: Dict[tuple, type] = {}

    def __init__(self, state_type: type):
        if state_type is None:
            raise TypeError("state_type is required and cannot be None")

        self._steps: List[Callable[..., Any]] = []
        self._memory: Optional[Any] = None
        self._built = False
        self._state_type = state_type
        self._node_names: List[str] = []
        self._has_chat_model: bool = False

    def add(
        self, step: Callable[..., Any], node_name: Optional[str] = None
    ) -> "AgentFactory":
        """
        Add an async step to the agent. If no node_name is provided, one will be auto-generated.
        """
        if node_name is None:
            node_name = f"step_{len(self._steps)}"

        self._steps.append(step)
        self._node_names.append(node_name)
        return self

    def with_memory(self, memory: Optional[Any] = None) -> "AgentFactory":
        """
        Add memory support to the agent.
        """
        self._memory = memory
        return self

    def with_chat(
        self, llm: Optional[Union[BaseChatModel, Dict[str, Any]]] = None, **kwargs
    ) -> "ChatStepBuilder":
        """
        Adds an LLM step to the chain. Returns a ChatStepBuilder for configuring this specific LLM step.
        """
        self._has_chat_model = True

        # Store the config for this LLM step
        config = None
        instance = None
        if llm is None:
            config = {"provider": "openai", **kwargs}
        elif isinstance(llm, BaseChatModel):
            instance = llm
        elif isinstance(llm, dict):
            config = llm
        else:
            raise ValueError("llm must be None, a BaseChatModel instance, or a dict")

        # Create LLMStep instance directly
        llm_step = LLMStep("", "", config, instance)

        node_name = f"llm_step_{len(self._steps)}"
        self._steps.append(llm_step)
        self._node_names.append(node_name)

        # Return a ChatStepBuilder that can configure this specific LLM step
        return ChatStepBuilder(self, len(self._steps) - 1)

    def _create_state_type(self) -> type:
        """
        Create the appropriate state type based on user input and chat model presence.
        """
        if self._has_chat_model:
            type_hints = get_type_hints(self._state_type, include_extras=True)
            if "messages" in type_hints:
                return self._state_type
            else:
                # Use a cache to avoid recreating the same combined type
                base_name = self._state_type.__name__
                dynamic_name = f"{base_name}WithMessagesAddedByPetal"
                cache_key = (base_name, tuple(sorted(type_hints.items())))
                if cache_key in self._dynamic_state_type_cache:
                    return self._dynamic_state_type_cache[cache_key]
                # Use multiple inheritance to combine user type and MessagesState
                combined_type = type(
                    dynamic_name, (self._state_type, MessagesState), {}
                )
                self._dynamic_state_type_cache[cache_key] = combined_type
                return combined_type
        else:
            return self._state_type

    def build(self) -> Agent:
        if not self._steps:
            raise RuntimeError("Cannot build Agent: no steps added")

        # Create the state type with messages field if needed
        final_state_type = self._create_state_type()

        # Create the StateGraph
        graph_builder: StateGraph = StateGraph(final_state_type)

        # Add nodes to the graph
        for i, (step, node_name) in enumerate(
            zip(self._steps, self._node_names, strict=False)
        ):
            # Add node to graph using node_name and step
            graph_builder.add_node(node_name, step)

            # Add edges to connect nodes in sequence
            if i == 0:
                # First node connects from START
                graph_builder.add_edge(START, node_name)
            else:
                # Connect to previous node
                prev_node = self._node_names[i - 1]
                graph_builder.add_edge(prev_node, node_name)

            # Last node connects to END
            if i == len(self._steps) - 1:
                graph_builder.add_edge(node_name, END)

        # Compile the graph
        compiled_graph = graph_builder.compile()

        # Create and return the agent
        agent = Agent().build(compiled_graph, final_state_type)
        self._built = True
        return agent


class ChatStepBuilder:
    """
    Builder for configuring a specific LLM step in the chain.
    """

    def __init__(self, factory: AgentFactory, step_index: int):
        self.factory = factory
        self.step_index = step_index

    def with_prompt(self, prompt_template: str) -> "ChatStepBuilder":
        """Set the prompt template for this specific LLM step."""
        step = self.factory._steps[self.step_index]
        if isinstance(step, LLMStep):
            step.prompt_template = prompt_template
        else:
            raise ValueError("Cannot set prompt on non-LLM step")
        return self

    def with_system_prompt(self, system_prompt: str) -> "ChatStepBuilder":
        """Set the system prompt for this specific LLM step."""
        step = self.factory._steps[self.step_index]
        if isinstance(step, LLMStep):
            step.system_prompt = system_prompt
        else:
            raise ValueError("Cannot set system prompt on non-LLM step")
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
