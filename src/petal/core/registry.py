import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool


class DiscoveryStrategy(ABC):
    """Abstract base for tool discovery strategies."""

    @abstractmethod
    async def discover(self, name: str) -> Optional[BaseTool]:
        """Discover a tool by name. Returns None if not found."""


class ToolRegistry:
    """Singleton tool registry with lazy discovery capabilities."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "ToolRegistry":
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry (only called once due to singleton)."""
        if not hasattr(self, "_initialized"):
            self._registry: Dict[str, BaseTool] = {}
            self._discovery_cache: Dict[str, bool] = {}  # Track failed discoveries
            self._discovery_chain: List[DiscoveryStrategy] = []
            self._initialized = True

    def add(self, name: str, tool: BaseTool) -> "ToolRegistry":
        """Register a tool by name. Must be a BaseTool instance."""
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"Tool '{name}' must be decorated with @tool or @petaltool and be a BaseTool instance. "
                f"Got type: {type(tool)}."
            )
        self._registry[name] = tool
        return self

    async def resolve(self, name: str) -> BaseTool:
        """Resolve tool with lazy discovery if not found."""
        # 1. Check direct registry
        if name in self._registry:
            return self._registry[name]

        # 2. Check if we've already tried to discover this
        if name in self._discovery_cache:
            raise KeyError(f"Tool '{name}' not found")

        # 3. Try discovery chain
        for strategy in self._discovery_chain:
            try:
                tool = await strategy.discover(name)
                if tool:
                    self._registry[name] = tool
                    return tool
            except Exception:
                continue

        # 4. Mark as not found
        self._discovery_cache[name] = False
        raise KeyError(f"Tool '{name}' not found")

    def list(self) -> List[str]:
        """List all registered tool names."""
        return sorted(self._registry.keys())

    def add_discovery_strategy(self, strategy: DiscoveryStrategy) -> "ToolRegistry":
        """Add a discovery strategy to the chain."""
        self._discovery_chain.append(strategy)
        return self

    def _reset_for_testing(self) -> None:
        """Reset the registry for testing purposes."""
        self._registry.clear()
        self._discovery_cache.clear()
        self._discovery_chain.clear()
