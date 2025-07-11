import operator
import re
from string import Formatter
from typing import Annotated, Any, Callable, Dict, List, Optional, Type

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from pydantic import BaseModel, Field
from rich.console import Console

from petal.core.tool_factory import ToolFactory

console = Console()

# Canonical ReAct instruction - unstructured output
REACT_INSTRUCTION = (
    "For each step, reason about what to do next by writing a 'Thought:' line, "
    "then, if needed, an 'Action:' line (with tool and arguments), and after a tool is called, "
    "an 'Observation:' line with the result. Repeat until you can answer the user's question."
)


class SafeFormatter(Formatter):
    """Safe formatter that handles missing keys gracefully."""

    def get_value(self, key, args, kwds):  # noqa: ARG002
        try:
            return kwds[key]
        except KeyError:
            console.print(
                f":warning: [bold yellow]No value found for placeholder '[red]{key}[/red]' in prompt, leaving as-is[/bold yellow] :warning:"
            )
            return f"{{{key}}}"


class ReActLoopState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    scratchpad: str = ""
    thoughts: Annotated[List[str], operator.add] = Field(default_factory=list)
    actions: Annotated[List[str], operator.add] = Field(default_factory=list)
    # Arbitrary context fields for prompt formatting
    context: Dict[str, Any] = Field(default_factory=dict)


class ReActAgentBuilder:
    """
    Standalone builder for a ReAct agent loop using an LLM and tools resolved from a ToolFactory.
    Accepts any user state schema. Internally uses a minimal ReActLoopState.
    At the end, calls an LLM to synthesize the final user state from accumulated information.
    """

    def __init__(
        self,
        state_schema: Type[BaseModel],
        llm: Optional[Any] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        tool_names: Optional[List[str]] = None,
        tool_factory: Optional[ToolFactory] = None,
        system_prompt: Optional[str] = None,
        structured_output_model: Optional[Type[BaseModel]] = None,
        next_node: Optional[str] = None,
    ):
        self.state_schema = state_schema
        self.tool_names = tool_names or []
        self.tool_factory = tool_factory or ToolFactory()
        self.structured_output_model = structured_output_model
        self.next_node = next_node
        self.safe_formatter = SafeFormatter()

        # Compose the system prompt
        base_prompt = (
            system_prompt.strip() if system_prompt else "You are a helpful assistant."
        )
        if REACT_INSTRUCTION not in base_prompt:
            base_prompt += "\n\n" + REACT_INSTRUCTION
        self.system_prompt = base_prompt

        # Construct or use provided LLM
        if llm is not None:
            self.llm = llm
        elif llm_config is not None:
            provider = llm_config.get("provider", "openai")
            if provider == "openai":
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(
                    **{k: v for k, v in llm_config.items() if k != "provider"}
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        else:
            raise ValueError("Must provide either llm or llm_config")

    def _extract_react_segments(
        self, ai_message_content: Any
    ) -> tuple[str, List[str], List[str]]:
        if not isinstance(ai_message_content, str):
            return "", [], []
        segments = []
        thoughts = []
        actions = []
        thought_matches = re.findall(
            r"Thought:(.*?)(?=\n|$)", ai_message_content, re.DOTALL | re.IGNORECASE
        )
        for match in thought_matches:
            thought = match.strip()
            if thought:
                thoughts.append(thought)
                segments.append(f"Thought: {thought}")
        action_matches = re.findall(
            r"Action:(.*?)(?=\n|$)", ai_message_content, re.DOTALL | re.IGNORECASE
        )
        for match in action_matches:
            action = match.strip()
            if action:
                actions.append(action)
                segments.append(f"Action: {action}")
        scratchpad_addition = "\n".join(segments)
        return scratchpad_addition, thoughts, actions

    def _safe_get_attr(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Safely get attribute from object, handling both dict and object access."""
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    def _safe_model_dump(self, obj: Any) -> Dict[str, Any]:
        """Safely convert object to dict, handling both Pydantic models and dicts."""
        if isinstance(obj, dict):
            return obj.copy()
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        # Fallback for other objects
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}

    def _user_to_internal(self, user_state: Any) -> ReActLoopState:
        """Convert user state to internal state, handling both dict and Pydantic models."""
        # Copy messages, scratchpad, thoughts, actions if present, else default
        messages = self._safe_get_attr(user_state, "messages", [])
        scratchpad = self._safe_get_attr(user_state, "scratchpad", "")
        thoughts = self._safe_get_attr(user_state, "thoughts", [])
        actions = self._safe_get_attr(user_state, "actions", [])

        # All other fields as context
        context = self._safe_model_dump(user_state)
        for k in ["messages", "scratchpad", "thoughts", "actions"]:
            context.pop(k, None)

        return ReActLoopState(
            messages=messages,
            scratchpad=scratchpad,
            thoughts=thoughts,
            actions=actions,
            context=context,
        )

    def _format_prompt_safely(self, template: str, context: Dict[str, Any]) -> str:
        """Format a prompt template safely using SafeFormatter."""
        try:
            return template.format(**context)
        except Exception:
            return self.safe_formatter.format(template, **context)

    async def _synthesize_final_state(
        self, internal_state: ReActLoopState, original_user_state: Any
    ) -> BaseModel:
        """
        Use LLM to synthesize final user state from accumulated information.

        Args:
            internal_state: The internal ReAct loop state
            original_user_state: The original user state (used for schema validation)
        """
        # Validate that we're using the correct schema
        if not isinstance(original_user_state, self.state_schema):
            console.print(
                f"[bold yellow]Warning: Original user state type {type(original_user_state)} doesn't match expected schema {self.state_schema}[/bold yellow]"
            )

        # Create a summary of all accumulated information
        conversation_summary = "\n".join(
            [
                f"Message {i+1}: {msg.content}"
                for i, msg in enumerate(internal_state.messages)
                if hasattr(msg, "content")
            ]
        )

        reasoning_summary = "\n".join(
            [f"Thought: {thought}" for thought in internal_state.thoughts]
        )

        actions_summary = "\n".join(
            [f"Action: {action}" for action in internal_state.actions]
        )

        scratchpad_summary = internal_state.scratchpad

        # Filter out LangGraph internal state from context
        clean_context = {}
        if internal_state.context:
            for key, value in internal_state.context.items():
                # Skip LangGraph internal keys (they start with __)
                # Only include simple, serializable values
                if (
                    not key.startswith("__")
                    and isinstance(value, (str, int, float, bool, list, dict))
                    and not str(type(value)).startswith("<class")
                ):
                    clean_context[key] = value

        # Create prompt for LLM to synthesize final state
        synthesis_prompt = f"""
Based on the following conversation and reasoning, synthesize the final state according to the schema: {self.state_schema.__name__}

CONVERSATION:
{conversation_summary}

REASONING:
{reasoning_summary}

ACTIONS TAKEN:
{actions_summary}

SCRATCHPAD:
{scratchpad_summary}

ORIGINAL CONTEXT:
{clean_context}

Please populate the final state with appropriate values based on this information.
Return the state as a JSON object matching the schema structure.
"""

        # Use LLM to generate final state with function calling
        llm_with_schema = self.llm.with_structured_output(
            self.state_schema, method="function_calling"
        )
        final_state = await llm_with_schema.ainvoke(synthesis_prompt)

        return final_state

    def _create_prompt_node(self, base_prompt: str):
        """Create the prompt node with safe formatting."""

        async def prompt_node(state: Any, config: RunnableConfig) -> dict:
            context = {
                **self._safe_get_attr(state, "context", {}),
                **config.get("configurable", {}),
            }
            formatted_messages = []
            for message in self._safe_get_attr(state, "messages", []):
                if message.__class__.__name__ in (
                    "HumanMessage",
                    "SystemMessage",
                ) and isinstance(message.content, str):
                    message.content = self._format_prompt_safely(
                        message.content, context
                    )
                formatted_messages.append(message)

            # Store the base prompt in context for potential use
            context["base_prompt"] = base_prompt

            return {
                "messages": formatted_messages,
                "scratchpad": self._safe_get_attr(state, "scratchpad", ""),
                "context": context,
            }

        return prompt_node

    def _create_llm_node(self, base_prompt: str):
        """Create the LLM node with safe formatting."""

        async def llm_node(state: Any, config: RunnableConfig) -> dict:
            context = {
                **self._safe_get_attr(state, "context", {}),
                **config.get("configurable", {}),
            }
            formatted_system_prompt = self._format_prompt_safely(base_prompt, context)
            system_msg = SystemMessage(content=formatted_system_prompt)
            messages_for_llm = [system_msg] + list(
                self._safe_get_attr(state, "messages", [])
            )
            ai_msg = await self.llm_with_tools.ainvoke(messages_for_llm)
            scratchpad_add, new_thoughts, new_actions = self._extract_react_segments(
                getattr(ai_msg, "content", "")
            )
            new_scratchpad = (
                self._safe_get_attr(state, "scratchpad", "")
                + ("\n" + scratchpad_add if scratchpad_add else "")
            ).strip()
            return {
                "messages": [ai_msg],
                "scratchpad": new_scratchpad,
                "thoughts": new_thoughts,
                "actions": new_actions,
                "context": context,
            }

        return llm_node

    def _create_tool_node(self, tool_node: ToolNode):
        """Create the tool node with observation handling."""

        async def tool_node_with_obs(state: Any, config: RunnableConfig) -> dict:
            result = await tool_node.ainvoke(state, config)
            tool_msgs = result.get("messages", [])
            obs_lines = []
            for msg in tool_msgs:
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    obs_lines.append(f"Observation: {msg.content}")
            obs_scratchpad = "\n".join(obs_lines)
            new_scratchpad = (
                self._safe_get_attr(state, "scratchpad", "")
                + ("\n" + obs_scratchpad if obs_scratchpad else "")
            ).strip()
            return {
                **result,
                "scratchpad": new_scratchpad,
                "context": self._safe_get_attr(state, "context", {}),
            }

        return tool_node_with_obs

    def _create_decide_next_step(self):
        """Create the decision function for next step."""

        def decide_next_step(state) -> str:
            messages = self._safe_get_attr(state, "messages", [])
            if not messages:
                return self.next_node or END
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return self.next_node or END

        return decide_next_step

    def build(self) -> Callable:
        # Resolve tools
        tools = [self.tool_factory.resolve(name) for name in self.tool_names]

        # Create tool documentation for the prompt
        tool_docs = []
        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "description"):
                tool_docs.append(f"- {tool.name}: {tool.description}")
            elif hasattr(tool, "__name__"):
                tool_docs.append(
                    f"- {tool.__name__}: {getattr(tool, '__doc__', 'No description')}"
                )

        tool_section = "\n".join(tool_docs) if tool_docs else "No tools available"

        # Bind tools to LLM if it supports it
        if hasattr(self.llm, "bind_tools"):
            self.llm_with_tools = self.llm.bind_tools(tools)
        else:
            self.llm_with_tools = self.llm

        # Compose the system prompt with tool information
        base_prompt = self.system_prompt.strip()

        # Add tool information
        if tool_docs:
            base_prompt += f"\n\nAvailable tools:\n{tool_section}"

        # Add ReAct instruction if not already present
        if REACT_INSTRUCTION not in base_prompt:
            base_prompt += "\n\n" + REACT_INSTRUCTION

        tool_node = ToolNode(tools)

        # Create nodes using helper methods
        prompt_node = self._create_prompt_node(base_prompt)
        llm_node = self._create_llm_node(base_prompt)
        tool_node_with_obs = self._create_tool_node(tool_node)
        decide_next_step = self._create_decide_next_step()

        graph = StateGraph(ReActLoopState)
        graph.add_node("prompt", prompt_node)
        graph.add_node("llm", llm_node)
        graph.add_node("tools", tool_node_with_obs)
        graph.set_entry_point("prompt")
        graph.add_edge("prompt", "llm")
        graph.add_conditional_edges("llm", decide_next_step)
        graph.add_edge("tools", "prompt")
        compiled_graph = graph.compile()

        async def arun_agent(user_state: Any, *args, **kwargs):
            internal_state = self._user_to_internal(user_state)
            result_dict = await compiled_graph.ainvoke(internal_state, *args, **kwargs)
            # Convert the result dict back to ReActLoopState
            result_state = ReActLoopState(**result_dict)
            return await self._synthesize_final_state(result_state, user_state)

        return arun_agent
