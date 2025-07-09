import asyncio
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints

from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class DefaultState(TypedDict):
    """Default state schema for agents."""

    messages: Annotated[list, add_messages]


class FlexibleState(TypedDict, total=False):
    """Flexible state schema that allows any keys while supporting message accumulation."""

    messages: Annotated[list, add_messages]


class MergeableState(TypedDict, total=False):
    """State schema that supports merging of any fields."""

    # This allows any keys to be added and merged
    # We'll use a simple dict approach for now


class Agent:
    """
    The runnable agent object, composed of a LangGraph StateGraph.
    """

    def __init__(self, graph: StateGraph, state_type: Optional[type] = None):
        self.graph = graph
        self.state_type = state_type or DefaultState
        self.built = True

    def run(self, state: dict) -> dict:
        if not getattr(self, "built", False):
            raise RuntimeError("Agent.run() called before build()")
        return self.graph.invoke(state)

    async def arun(self, state: dict) -> dict:
        if not getattr(self, "built", False):
            raise RuntimeError("Agent.arun() called before build()")
        return await self.graph.ainvoke(state)


class AgentFactory:
    """
    Builder and fluent interface for constructing Agent objects as LangGraph StateGraphs.
    """

    def __init__(self, state_type: type):
        self._steps: List[Callable[[Any], Any]] = []
        self._memory: Optional[Any] = None
        self._built = False
        self._state_type = state_type
        self._node_names: List[str] = []
        self._has_chat_model = False

    def add(
        self, step: Callable[[Any], Any], node_name: Optional[str] = None
    ) -> "AgentFactory":
        """
        Add a step to the agent. If no node_name is provided, one will be auto-generated.
        """
        if node_name is None:
            node_name = f"step_{len(self._steps)}"

        self._steps.append(step)
        self._node_names.append(node_name)
        return self

    def with_memory(self, memory: Optional[Any] = None, **kwargs) -> "AgentFactory":
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

        # Create a placeholder step that will be replaced in build()
        def _llm_step_placeholder(state):
            raise RuntimeError("LLM step not built yet")

        _llm_step_placeholder._llm_config = config
        _llm_step_placeholder._llm_instance = instance
        _llm_step_placeholder._prompt_template = ""
        _llm_step_placeholder._system_prompt = ""

        node_name = f"llm_step_{len(self._steps)}"
        self._steps.append(_llm_step_placeholder)
        self._node_names.append(node_name)

        # Return a ChatStepBuilder that can configure this specific LLM step
        return ChatStepBuilder(self, len(self._steps) - 1)

    def _create_state_type(self) -> type:
        """
        Create the state type, adding messages field with add_messages reducer if needed.
        For non-chat agents, use the provided state_type or dict.
        """
        if self._state_type is not None:
            # User provided a custom state type
            if self._has_chat_model:
                # Check if the state type already has a messages field
                type_hints = get_type_hints(self._state_type, include_extras=True)
                has_messages = "messages" in type_hints
                if has_messages:
                    # State type already has messages, use as-is
                    return self._state_type

                # Need to add messages field with add_messages reducer
                # Create a new TypedDict that inherits from the original
                class DynamicState(self._state_type):
                    messages: Annotated[list, add_messages]

                return DynamicState
            else:
                # No chat model, use the provided state type as-is
                return self._state_type
        else:
            # No state type provided
            if self._has_chat_model:
                # Use flexible state with messages
                return FlexibleState
            else:
                # For non-chat agents, create a flexible state type that allows any fields
                # This allows partial updates to work properly
                class FlexibleNonChatState(TypedDict, total=False):
                    # Allow any fields to be added dynamically
                    pass

                return FlexibleNonChatState

    def build(self) -> Agent:
        if not self._steps:
            raise RuntimeError("Cannot build Agent: no steps added")

        # Create the state type with messages field if needed
        final_state_type = self._create_state_type()

        # Create the StateGraph
        graph_builder = StateGraph(final_state_type)

        # Add nodes to the graph
        for i, (step, node_name) in enumerate(zip(self._steps, self._node_names, strict=False)):
            if hasattr(step, "_llm_config") or hasattr(step, "_llm_instance"):
                # Build LLM step
                built_step = self._make_llm_step(
                    getattr(step, "_prompt_template", ""),
                    getattr(step, "_system_prompt", ""),
                    getattr(step, "_llm_config", None),
                    getattr(step, "_llm_instance", None),
                )
            else:
                built_step = step

            # Add node to graph
            graph_builder.add_node(node_name, built_step)

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
        agent = Agent(compiled_graph, final_state_type)
        self._built = True
        return agent

    @staticmethod
    def _make_memory_load_step(memory):
        def load_memory(state: Dict) -> Dict:
            # Load memory variables and update state
            mem_vars = memory.load_memory_variables(state)
            state.update(mem_vars)
            return state

        return load_memory

    @staticmethod
    def _make_llm_step(prompt_template, system_prompt, llm_config, llm_instance):
        def llm_step(state):
            llm = llm_instance
            if llm is None:
                config = llm_config or {}
                provider = config.get("provider", "openai")
                openai_config = {k: v for k, v in config.items() if k != "provider"}
                if "model" not in openai_config:
                    openai_config["model"] = "gpt-4o-mini"
                if "temperature" not in openai_config:
                    openai_config["temperature"] = 0
                if provider == "openai":
                    llm = ChatOpenAI(**openai_config)
                else:
                    raise ValueError(f"Unsupported LLM provider: {provider}")

            # Get the original messages
            original_messages = state.get("messages", [])

            # Build the messages for LLM invocation (temporary)
            llm_messages = []

            # Add system prompt at the top if provided
            if system_prompt:
                llm_messages.append({"role": "system", "content": system_prompt})

            # Add original messages
            llm_messages.extend(original_messages)

            # Add user prompt at the end if provided
            if prompt_template:
                user_prompt = prompt_template.format(**state)
                llm_messages.append({"role": "user", "content": user_prompt})

            # Get response from LLM
            response = llm.invoke(llm_messages)

            # Return user prompt + response - LangGraph will merge it using add_messages reducer
            if prompt_template:
                user_prompt = prompt_template.format(**state)
                return {
                    "messages": [{"role": "user", "content": user_prompt}, response]
                }
            else:
                return {"messages": [response]}

        async def llm_step_async(state):
            llm = llm_instance
            if llm is None:
                config = llm_config or {}
                provider = config.get("provider", "openai")
                openai_config = {k: v for k, v in config.items() if k != "provider"}
                if "model" not in openai_config:
                    openai_config["model"] = "gpt-4o-mini"
                if "temperature" not in openai_config:
                    openai_config["temperature"] = 0
                if provider == "openai":
                    llm = ChatOpenAI(**openai_config)
                else:
                    raise ValueError(f"Unsupported LLM provider: {provider}")

            # Get the original messages
            original_messages = state.get("messages", [])

            # Build the messages for LLM invocation (temporary)
            llm_messages = []

            # Add system prompt at the top if provided
            if system_prompt:
                llm_messages.append({"role": "system", "content": system_prompt})

            # Add original messages
            llm_messages.extend(original_messages)

            # Add user prompt at the end if provided
            if prompt_template:
                user_prompt = prompt_template.format(**state)
                llm_messages.append({"role": "user", "content": user_prompt})

            # Get response from LLM
            response = await llm.ainvoke(llm_messages)

            # Return user prompt + response - LangGraph will merge it using add_messages reducer
            if prompt_template:
                user_prompt = prompt_template.format(**state)
                return {
                    "messages": [{"role": "user", "content": user_prompt}, response]
                }
            else:
                return {"messages": [response]}

        # Return a function that dispatches sync/async
        def step(state):
            if asyncio.iscoroutinefunction(llm_step_async) and getattr(
                state, "__async__", False
            ):
                return llm_step_async(state)
            return llm_step(state)

        step._is_llm_step = True
        step._llm_sync = llm_step
        step._llm_async = llm_step_async
        return step


class ChatStepBuilder:
    """
    Builder for configuring a specific LLM step in the chain.
    """

    def __init__(self, factory: AgentFactory, step_index: int):
        self.factory = factory
        self.step_index = step_index

    def with_prompt(self, prompt_template: str) -> "ChatStepBuilder":
        """Set the prompt template for this specific LLM step."""
        self.factory._steps[self.step_index]._prompt_template = prompt_template
        return self

    def with_system_prompt(self, system_prompt: str) -> "ChatStepBuilder":
        """Set the system prompt for this specific LLM step."""
        self.factory._steps[self.step_index]._system_prompt = system_prompt
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
