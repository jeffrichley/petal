"""Folder-based tool discovery strategy."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Dict, List, Optional, Set

from langchain_core.tools import BaseTool

from petal.core.discovery.module_cache import ModuleCache
from petal.core.registry import DiscoveryStrategy

DEFAULT_TOOL_FOLDERS = ["tools/", "src/tools/", "app/tools/", "lib/tools/"]


class FolderDiscovery(DiscoveryStrategy):
    """Discovers tools by scanning project folders."""

    def __init__(
        self,
        folders: Optional[List[str]] = None,
        module_cache: Optional[ModuleCache] = None,
    ) -> None:
        self.folders = folders or DEFAULT_TOOL_FOLDERS
        self.module_cache = module_cache or ModuleCache()
        self._scanned_folders: Set[str] = set()
        self._folder_tools: Dict[str, Dict[str, BaseTool]] = {}

    async def discover(self, name: str) -> Optional[BaseTool]:
        for folder in self.folders:
            if folder not in self._scanned_folders:
                await self._scan_folder(folder)
            tools = self._folder_tools.get(folder, {})
            if name in tools:
                return tools[name]
        return None

    async def _scan_folder(self, folder: str) -> None:
        self._scanned_folders.add(folder)
        self._folder_tools.setdefault(folder, {})

        if not os.path.isdir(folder):
            return

        for root, _dirs, files in os.walk(folder):
            for file in files:
                if not file.endswith(".py"):
                    continue
                file_path = os.path.join(root, file)
                module_name = self._import_module_from_path(file_path)
                if module_name:
                    try:
                        tools = await self.module_cache.scan_module(module_name)
                        self._folder_tools[folder].update(tools)
                    except Exception:
                        continue

    def _import_module_from_path(self, file_path: str) -> Optional[str]:
        module_name = f"folder_{abs(hash(file_path))}"
        if module_name in sys.modules:
            return module_name
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module_name
        except Exception:
            return None
