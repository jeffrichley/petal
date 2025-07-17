"""React step strategy for creating ReAct agent loops."""

from typing import Any, Callable, Dict

from langchain_core.runnables import RunnableConfig

from petal.core.builders.react import ReActAgentBuilder
from petal.core.steps.base import StepStrategy
from petal.core.tool_factory import ToolFactory


class ReactStepStrategy(StepStrategy):
    """Strategy for creating React reasoning loop steps."""

    async def create_step(self, config: Dict[str, Any]) -> Callable:
        """Create a React step from configuration.

        Args:
            config: Configuration dictionary containing:
                - tools: List of tool names or tool objects
                - llm_instance: LLM instance (optional)
                - llm_config: LLM configuration (optional)
                - system_prompt: System prompt (optional)
                - prompt_template: Prompt template (optional)
                - structured_output_model: Pydantic model for output (optional)
                - state_schema: User state schema (required)

        Returns:
            The callable React step function.

        Raises:
            ValueError: If required configuration is missing or invalid.
        """
        # Extract configuration
        tools = config.get("tools", [])
        llm_instance = config.get("llm_instance")
        llm_config = config.get("llm_config")
        system_prompt = config.get("system_prompt")
        structured_output_model = config.get("structured_output_model")
        state_schema = config.get("state_schema")

        if not tools:
            raise ValueError("React steps require at least one tool")

        if not state_schema:
            raise ValueError("React steps require a state_schema")

        # Determine if tools are strings or objects
        tool_names = None
        tool_factory = config.get("tool_factory", ToolFactory())

        if tools and isinstance(tools[0], str):
            tool_names = tools
        elif tools:
            # For tool objects, we need to handle them differently
            # For now, let's convert them to names for simplicity
            tool_names = [
                getattr(tool, "name", f"tool_{i}") for i, tool in enumerate(tools)
            ]

        # Provide default LLM config if none provided
        if llm_instance is None and llm_config is None:
            llm_config = {"provider": "openai", "model": "gpt-4o-mini"}

        # Create ReActAgentBuilder
        react_builder = ReActAgentBuilder(
            state_schema=state_schema,
            llm=llm_instance,
            llm_config=llm_config,
            tool_names=tool_names,
            tool_factory=tool_factory,
            system_prompt=system_prompt,
            structured_output_model=structured_output_model,
        )

        # Build the React agent
        react_agent = react_builder.build()

        # Create the step function
        async def react_step(
            state: Any,
            config: RunnableConfig,  # noqa: ARG001
        ) -> Dict[str, Any]:
            """React step function that handles state conversion and execution."""
            # Run the React agent
            result = await react_agent(state)

            # Return the result
            return result

        return react_step

    def get_node_name(self, index: int) -> str:
        """Generate node name for React step."""
        return f"react_step_{index}"
