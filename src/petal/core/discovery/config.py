"""Configuration-based tool discovery strategy."""

import os
from typing import Dict, List, Optional

import yaml
from langchain_core.tools import BaseTool

from petal.core.registry import DiscoveryStrategy

DEFAULT_CONFIG_LOCATIONS = [
    "configs/tools.yaml",
    "configs/tools.yml",
    "tools.yaml",
    "tools.yml",
    "configs/",
    "config/",
    "conf/",
]


class ConfigDiscovery(DiscoveryStrategy):
    """Discovers tools from YAML configuration files."""

    def __init__(self, config_locations: Optional[List[str]] = None) -> None:
        """Initialize the config discovery strategy."""
        self.config_locations = config_locations or DEFAULT_CONFIG_LOCATIONS
        self._config_cache: Dict[str, Dict] = {}

    async def discover(self, name: str) -> Optional[BaseTool]:
        """Discover a tool by scanning YAML configuration files."""
        try:
            # Load config files if not cached
            if not self._config_cache:
                await self._load_config_files()

            # Look for the tool in the cached configs
            for config in self._config_cache.values():
                if "tools" in config and name in config["tools"]:
                    tool_config = config["tools"][name]
                    return await self._load_tool_from_config(name, tool_config)

            return None
        except ValueError:
            # Re-raise ValueError (e.g., malformed YAML)
            raise
        except Exception:
            # Gracefully handle other discovery errors
            return None

    async def _load_config_files(self) -> None:
        """Load and cache configuration files."""
        for location in self.config_locations:
            try:
                if os.path.isfile(location):
                    config = await self._load_config_from_file(location)
                    if config:
                        self._config_cache[location] = config
                elif os.path.isdir(location):
                    # Scan directory for YAML files
                    for filename in os.listdir(location):
                        if filename.endswith((".yaml", ".yml")):
                            file_path = os.path.join(location, filename)
                            config = await self._load_config_from_file(file_path)
                            if config:
                                self._config_cache[file_path] = config
            except ValueError:
                # Re-raise ValueError from YAML parsing
                raise
            except Exception:
                # Skip other invalid config files
                continue

    async def _load_config_from_file(self, file_path: str) -> Optional[Dict]:
        """Load configuration from a single file."""
        try:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
                return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}") from e
        except Exception:
            return None

    async def _load_tool_from_config(self, name: str, tool_config: Dict) -> BaseTool:
        """Load a tool from its configuration."""
        module_name = tool_config.get("module")
        function_name = tool_config.get("function")

        if not module_name or not function_name:
            raise ValueError(f"Tool '{name}' config missing module or function")

        # Import the module dynamically
        try:
            module = __import__(module_name, fromlist=[function_name])
            function = getattr(module, function_name)

            # If the function is already a BaseTool, return it
            if isinstance(function, BaseTool):
                # Create a copy with the correct name using model_copy for Pydantic v2
                tool = function.model_copy()
                tool.name = name
                return tool

            # If it's a regular function, we need to wrap it
            # For now, assume it's already decorated with @tool or @petaltool
            # This is a simplified implementation
            return function

        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Could not load tool '{name}' from {module_name}.{function_name}: {e}"
            ) from e
