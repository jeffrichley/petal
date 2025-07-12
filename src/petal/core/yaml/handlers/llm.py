"""LLM node configuration handler."""

from typing import Callable

from petal.core.config.yaml import BaseNodeConfig, LLMNodeConfig
from petal.core.steps.registry import StepRegistry
from petal.core.yaml.handlers.base import NodeConfigHandler


class LLMNodeHandler(NodeConfigHandler):
    """Handler for creating LLM nodes from YAML configuration."""

    def __init__(self):
        self.registry = StepRegistry()

    def create_node(self, config: BaseNodeConfig) -> Callable:
        """Create an LLM node from configuration.

        Args:
            config: The LLM node configuration

        Returns:
            A callable LLM node function
        """
        # Cast config to LLMNodeConfig
        llm_config = (
            config
            if isinstance(config, LLMNodeConfig)
            else LLMNodeConfig(**config.__dict__)
        )
        # Convert config to step config format
        step_config = self._config_to_step_config(llm_config)
        return self.registry.create_step(step_config)

    def _config_to_step_config(self, config: LLMNodeConfig):
        """Convert LLMNodeConfig to StepConfig format."""
        from petal.core.config.agent import StepConfig

        # Convert LLM config to step config
        step_config_dict = {
            "provider": config.provider,
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

        if config.prompt:
            step_config_dict["prompt_template"] = config.prompt

        if config.system_prompt:
            step_config_dict["system_prompt"] = config.system_prompt

        return StepConfig(
            strategy_type="llm", node_name=config.name, config=step_config_dict
        )
