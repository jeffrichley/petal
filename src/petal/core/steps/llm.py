from typing import Any, Dict, Optional

from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI

from petal.core.steps.base import StepStrategy


class LLMStep:
    """
    Encapsulates the configuration and logic for an LLM step.
    """

    def __init__(
        self,
        prompt_template: str,
        system_prompt: str,
        llm_config: Optional[Dict[str, Any]],
        llm_instance: Optional[BaseChatModel],
    ):
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.llm_config = llm_config
        self.llm_instance = llm_instance

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        llm = self._create_llm_instance()
        llm_messages, user_prompt = self._build_llm_messages(state)
        response = await llm.ainvoke(llm_messages)
        return self._format_llm_response(response, user_prompt)

    def _build_llm_messages(self, state: Dict[str, Any]):
        original_messages = state.get("messages", [])
        llm_messages = []
        if self.system_prompt:
            try:
                formatted_system_prompt = self.system_prompt.format(**state)
            except KeyError as e:
                missing_key = str(e).strip("'")
                raise ValueError(
                    f"System prompt template '{self.system_prompt}' requires key '{missing_key}' "
                    f"but it's not available in the state. Available keys: {list(state.keys())}"
                ) from e
            llm_messages.append({"role": "system", "content": formatted_system_prompt})
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

        # Filter out non-LLM parameters that shouldn't be passed to the LLM instance
        llm_params = {
            k: v
            for k, v in config.items()
            if k not in ["provider", "prompt_template", "system_prompt"]
        }

        if "model" not in llm_params:
            llm_params["model"] = "gpt-4o-mini"
        if "temperature" not in llm_params:
            llm_params["temperature"] = 0
        if provider == "openai":
            return ChatOpenAI(**llm_params)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _format_llm_response(self, response, user_prompt):
        if user_prompt:
            return {"messages": [{"role": "user", "content": user_prompt}, response]}
        else:
            return {"messages": [response]}


class LLMStepStrategy(StepStrategy):
    """
    Strategy for creating LLM steps.
    """

    def create_step(self, config: Dict[str, Any]) -> LLMStep:
        prompt_template = config.get("prompt_template", "")
        system_prompt = config.get("system_prompt", "")

        # Handle different config formats for backward compatibility
        llm_config = None
        llm_instance = None

        # Check for llm_instance first (direct instance)
        if "llm_instance" in config:
            llm_instance = config["llm_instance"]
        # Check for llm_config (separate config dict)
        elif "llm_config" in config:
            llm_config = config["llm_config"]
        # Check for provider-based config (new format)
        elif "provider" in config or any(
            key in config for key in ["model", "temperature", "api_key"]
        ):
            llm_config = config
        # If no LLM config provided, use default OpenAI config
        else:
            llm_config = {"provider": "openai", "model": "gpt-4o-mini"}

        return LLMStep(prompt_template, system_prompt, llm_config, llm_instance)

    def get_node_name(self, index: int) -> str:
        return f"llm_step_{index}"
