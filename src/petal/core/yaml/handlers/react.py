"""React node configuration handler."""

from typing import Any, Callable, Dict

from petal.core.config.yaml import BaseNodeConfig, ReactNodeConfig
from petal.core.steps.registry import StepRegistry
from petal.core.tool_factory import ToolFactory
from petal.core.yaml.handlers.base import NodeConfigHandler


class ReactNodeHandler(NodeConfigHandler):
    """Handler for creating React nodes from YAML configuration."""

    def __init__(self, tool_factory: Any = None):
        self.tool_factory = tool_factory or ToolFactory()
        self.registry = StepRegistry()

    def _register_mcp_servers(self, mcp_servers: Dict[str, Any]):
        """Register all MCP servers from the config with the ToolFactory using the generic resolver."""
        for server_name, server_info in mcp_servers.items():
            mcp_config = server_info.get("config")
            if mcp_config:
                self.tool_factory.add_mcp(server_name, mcp_config=mcp_config)

    def create_node(self, config: BaseNodeConfig) -> Callable:
        """Create a React node from configuration.

        Args:
            config: The React node configuration

        Returns:
            A callable React node function
        """
        # Cast config to ReactNodeConfig
        react_config = (
            config
            if isinstance(config, ReactNodeConfig)
            else ReactNodeConfig(**config.__dict__)
        )
        # Register MCP servers if present
        mcp_servers = getattr(react_config, "mcp_servers", None)
        if mcp_servers:
            self._register_mcp_servers(mcp_servers)
        # Resolve tools using ToolFactory
        resolved_tools = [
            self.tool_factory.resolve(tool) for tool in react_config.tools
        ]
        # Convert config to step config format
        step_config = self._config_to_step_config(react_config, resolved_tools)
        return self.registry.create_step(step_config)

    def _config_to_step_config(self, config: ReactNodeConfig, resolved_tools):
        """Convert ReactNodeConfig to StepConfig format."""
        from petal.core.config.agent import StepConfig

        # Provide a dummy state_schema for testability
        dummy_state_schema = object

        # Convert React config to step config
        step_config_dict = {
            "tools": resolved_tools,
            "max_iterations": config.max_iterations,
            "state_schema": dummy_state_schema,
        }

        if config.reasoning_prompt:
            step_config_dict["reasoning_prompt"] = config.reasoning_prompt

        if config.system_prompt:
            step_config_dict["system_prompt"] = config.system_prompt

        return StepConfig(
            strategy_type="react", node_name=config.name, config=step_config_dict
        )
