"""Decorator-based tool discovery strategy."""

from typing import Optional

from langchain_core.tools import BaseTool

from petal.core.discovery.module_cache import ModuleCache
from petal.core.registry import DiscoveryStrategy


class DecoratorDiscovery(DiscoveryStrategy):
    """Discovers tools decorated with @petaltool."""

    def __init__(self, module_cache: Optional[ModuleCache] = None) -> None:
        """Initialize the decorator discovery strategy."""
        self.module_cache = module_cache or ModuleCache()

    def _find_tools_by_base_name(self, all_tools: dict, base_name: str) -> list:
        """Finds all tools that match a base name (after the last colon)."""
        matching_tools = []
        for tool_name in all_tools:
            # Extract base name (after last colon)
            tool_base_name = tool_name.split(":")[-1] if ":" in tool_name else tool_name

            if tool_base_name == base_name:
                matching_tools.append(tool_name)

        return matching_tools

    async def discover(self, name: str) -> Optional[BaseTool]:
        """Discover a tool by scanning for @petaltool decorated functions."""
        try:
            # Scan all modules for decorated tools
            all_tools = await self.module_cache.scan_all_modules()

            # 1. Check direct match first
            if name in all_tools:
                return all_tools[name]

            # 2. Check if this might be a base name (no colons)
            if ":" not in name:
                matching_tools = self._find_tools_by_base_name(all_tools, name)
                if len(matching_tools) == 1:
                    # Single match - return it
                    return all_tools[matching_tools[0]]
                elif len(matching_tools) > 1:
                    # Multiple matches - return None (let ToolRegistry handle ambiguity)
                    return None

            # 3. Not found
            return None
        except Exception:
            # Gracefully handle any discovery errors
            return None
