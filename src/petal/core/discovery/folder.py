"""Folder-based tool discovery strategy."""

import fnmatch
import sys
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool

from petal.core.discovery.module_cache import ModuleCache
from petal.core.registry import DiscoveryStrategy


class FolderDiscovery(DiscoveryStrategy):
    """Discovers tools by scanning folders for Python files with zero-config defaults."""

    # Default folders to scan (in order of preference)
    DEFAULT_TOOL_FOLDERS = [
        "tools/",
        "src/tools/",
        "app/tools/",
        "lib/tools/",
        "petal_tools/",
        "custom_tools/",
    ]

    # Default exclusion patterns
    DEFAULT_EXCLUDE_PATTERNS = [
        "test_*",
        "*_test.py",
        "temp_*",
        "*_backup.py",
        "__pycache__",
        ".git",
        ".venv",
        "venv",
    ]

    # Python path scanning patterns
    PYTHON_PATH_TOOL_PATTERNS = ["*tool*.py", "*tools*.py", "*_tool.py", "*_tools.py"]

    def __init__(
        self,
        auto_discover: bool = True,
        folders: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        scan_python_path: bool = True,
    ) -> None:
        """
        Initialize folder discovery strategy with zero-config defaults.

        Args:
            auto_discover: If True, automatically discover all tools on first use.
            folders: List of folder paths to scan. If None, uses defaults.
            exclude_patterns: List of glob patterns to exclude.
            recursive: Whether to scan subdirectories recursively.
            scan_python_path: Whether to scan Python path for tools.
        """
        self.auto_discover = auto_discover
        self.folders = folders or self.DEFAULT_TOOL_FOLDERS.copy()
        self.exclude_patterns = exclude_patterns or self.DEFAULT_EXCLUDE_PATTERNS.copy()
        self.recursive = recursive
        self.scan_python_path = scan_python_path

        # Internal state
        self._scanned = False
        self._cached_tools: Dict[str, BaseTool] = {}
        self._module_cache = ModuleCache()

    async def discover(self, name: str) -> Optional[BaseTool]:
        """
        Discover a tool by scanning configured folders.
        If auto_discover=True, performs full scan on first call.

        Args:
            name: Tool name to discover.

        Returns:
            BaseTool if found, None otherwise.
        """
        # If auto_discover is enabled and we haven't scanned yet, do full scan
        if (
            self.auto_discover
            and not self._scanned
            or not self.auto_discover
            and not self._scanned
        ):
            await self._perform_full_scan()

        # Return from cache if found
        return self._cached_tools.get(name)

    async def discover_all(self) -> Dict[str, BaseTool]:
        """
        Discover all tools in configured folders and Python path.

        Returns:
            Dict mapping tool names to BaseTool instances.
        """
        if not self._scanned:
            await self._perform_full_scan()
        return self._cached_tools.copy()

    def add_folder(self, folder: str) -> "FolderDiscovery":
        """Add a folder to scan."""
        if folder not in self.folders:
            self.folders.append(folder)
        return self

    def add_exclude_pattern(self, pattern: str) -> "FolderDiscovery":
        """Add an exclusion pattern."""
        if pattern not in self.exclude_patterns:
            self.exclude_patterns.append(pattern)
        return self

    def enable_auto_discovery(self) -> "FolderDiscovery":
        """Enable automatic discovery of all tools."""
        self.auto_discover = True
        return self

    def disable_auto_discovery(self) -> "FolderDiscovery":
        """Disable automatic discovery (manual control only)."""
        self.auto_discover = False
        return self

    def is_auto_discover_enabled(self) -> bool:
        """Return whether auto discovery is enabled."""
        return self.auto_discover

    async def _perform_full_scan(self) -> None:
        """Perform full scan of all configured locations."""
        self._cached_tools.clear()

        # Scan configured folders
        folder_tools = await self._scan_folders()
        self._cached_tools.update(folder_tools)

        # Scan Python path if enabled
        if self.scan_python_path:
            path_tools = await self._scan_python_path()
            self._cached_tools.update(path_tools)

        self._scanned = True

    async def _scan_folders(self) -> Dict[str, BaseTool]:
        """Scan configured folders for tools."""
        tools = {}

        for folder in self.folders:
            try:
                folder_path = Path(folder)
                if not folder_path.exists():
                    continue

                # Find Python files in folder
                pattern = "**/*.py" if self.recursive else "*.py"
                python_files = folder_path.glob(pattern)

                for file_path in python_files:
                    # Check if file should be excluded
                    if self._should_exclude_file(file_path):
                        continue

                    # Try to load tools from file
                    file_tools = await self._load_tools_from_file(file_path)
                    tools.update(file_tools)

            except Exception:
                # Gracefully handle any folder scanning errors
                continue

        return tools

    async def _scan_python_path(self) -> Dict[str, BaseTool]:
        """Scan Python path for tool modules."""
        tools = {}

        for path in sys.path:
            try:
                path_obj = Path(path)
                if not path_obj.exists() or not path_obj.is_dir():
                    continue

                # Look for tool-related files
                for pattern in self.PYTHON_PATH_TOOL_PATTERNS:
                    for file_path in path_obj.glob(pattern):
                        if self._should_exclude_file(file_path):
                            continue

                        file_tools = await self._load_tools_from_file(file_path)
                        tools.update(file_tools)

            except Exception:
                # Gracefully handle any path scanning errors
                continue

        return tools

    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if a file should be excluded based on patterns."""
        file_name = file_path.name

        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(file_name, pattern):
                return True

            # Also check if any parent directory matches
            for parent in file_path.parents:
                if fnmatch.fnmatch(parent.name, pattern):
                    return True

        return False

    async def _load_tools_from_file(self, file_path: Path) -> Dict[str, BaseTool]:
        import importlib
        import sys
        from pathlib import Path

        tools = {}

        try:
            # Ensure project root is in sys.path
            project_root = str(Path(__file__).resolve().parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # Patch sys.modules to ensure singleton
            import petal.core.registry as main_registry_mod

            sys.modules["petal.core.registry"] = main_registry_mod

            # Get registry state before import
            registry = main_registry_mod.ToolRegistry()
            tools_before = set(registry.list())

            # Import the module
            module_name = f"discovered_{file_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                module.__package__ = "petal.core.discovery"  # or closest package
                module.__loader__ = spec.loader
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Scan the module's namespace for BaseTool instances
                from langchain_core.tools import BaseTool

                for _name, obj in vars(module).items():
                    if isinstance(obj, BaseTool):
                        tools[obj.name] = obj

                if not tools:
                    raise ValueError(
                        f"No tools found in '{file_path.name}'. "
                        "If you are using @petaltool, make sure your function has a docstring or provide a description. "
                        "See https://python.langchain.com/docs/modules/agents/tools/custom_tools/#the-tool-decorator for details."
                    )

            # Find newly registered tools (registry diff)
            tools_after = set(registry.list())
            new_tools = tools_after - tools_before
            for tool_name in new_tools:
                try:
                    tool = await registry.resolve(tool_name)
                    tools[tool_name] = tool
                except Exception:
                    continue

        except Exception as e:
            # Gracefully handle any module loading errors
            # If it's our custom error, re-raise
            if isinstance(e, ValueError) and "No tools found" in str(e):
                raise
            pass

        return tools
