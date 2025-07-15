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

    async def discover(self, name: str) -> Optional[BaseTool]:
        """Discover a tool by scanning for @petaltool decorated functions."""
        try:
            # Scan all modules for decorated tools
            all_tools = await self.module_cache.scan_all_modules()

            # Return the tool if found
            return all_tools.get(name)
        except Exception:
            # Gracefully handle any discovery errors
            return None
