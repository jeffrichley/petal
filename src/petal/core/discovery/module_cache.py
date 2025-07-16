"""Module cache for tool discovery."""

import sys
import threading
from typing import Any, Dict, Optional, Set

from langchain_core.tools import BaseTool


class ModuleCache:
    """Caches module scans to avoid repeated imports."""

    def __init__(self) -> None:
        """Initialize the module cache."""
        self._scanned_modules: Set[str] = set()
        self._module_tools: Dict[str, Dict[str, BaseTool]] = {}
        self._lock = threading.Lock()

    async def scan_module(self, module_name: str) -> Dict[str, BaseTool]:
        """Scan a module for decorated tools and cache results."""
        with self._lock:
            if module_name in self._scanned_modules:
                return self._module_tools.get(module_name, {})

            tools = await self._scan_module_internal(module_name)
            self._module_tools[module_name] = tools
            self._scanned_modules.add(module_name)
            return tools

    async def scan_all_modules(self) -> Dict[str, BaseTool]:
        """Scan all imported modules for decorated tools."""
        all_tools: Dict[str, BaseTool] = {}

        # Get all loaded modules
        module_names = list(sys.modules.keys())

        for module_name in module_names:
            try:
                module_tools = await self.scan_module(module_name)
                all_tools.update(module_tools)
            except Exception:
                # Skip modules that can't be scanned
                continue

        return all_tools

    async def _scan_module_internal(self, module_name: str) -> Dict[str, BaseTool]:
        """Internal method to scan a single module for decorated tools."""
        tools: Dict[str, BaseTool] = {}

        try:
            module = sys.modules.get(module_name)
            if not module:
                return tools

            # Scan module members for decorated functions using vars()
            # This avoids deprecation warnings from getattr()
            module_vars = vars(module)
            for name, obj in module_vars.items():
                # Skip private attributes and built-ins
                if name.startswith("_"):
                    continue

                if self._is_decorated_tool(obj):
                    tool_name = self._extract_tool_name(obj, name)
                    if tool_name:
                        tools[tool_name] = obj

        except Exception:
            # Gracefully handle any module scanning errors
            pass

        return tools

    def _is_decorated_tool(self, obj: Any) -> bool:
        """Check if an object is a decorated tool."""
        # Check if it's a BaseTool instance (from @petaltool or @tool)
        if isinstance(obj, BaseTool):
            return True

        # Legacy check for _petaltool_metadata (for backward compatibility)
        return callable(obj) and hasattr(obj, "_petaltool_metadata")

    def _extract_tool_name(self, obj: Any, default_name: str) -> Optional[str]:
        """Extract the tool name from a decorated function."""
        # For BaseTool instances (from @petaltool or @tool)
        if isinstance(obj, BaseTool):
            return obj.name

        # Legacy check for _petaltool_metadata
        if hasattr(obj, "_petaltool_metadata"):
            metadata = obj._petaltool_metadata
            # Handle case where metadata exists but is empty or doesn't have 'name'
            if not metadata or not isinstance(metadata, dict):
                return default_name
            return metadata.get("name", default_name)

        return None

    def clear_cache(self) -> None:
        """Clear the module cache for testing."""
        with self._lock:
            self._scanned_modules.clear()
            self._module_tools.clear()
