# All tools registered with ToolFactory must be async-friendly (either sync or async functions).
import asyncio
from typing import Any, Callable, Coroutine, Dict, List

from langchain_core.tools import BaseTool


class ToolFactory:
    """
    Async-friendly registry for callable tools (sync or async).
    """

    def __init__(self) -> None:
        """
        Initialize the tool registry.
        """
        self._registry: Dict[str, Callable[..., Any]] = {}
        self._mcp_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._mcp_loaded: Dict[str, asyncio.Event] = {}

    def add(self, name: str, fn: BaseTool) -> "ToolFactory":
        """
        Register a tool by name. Must be a BaseTool instance (decorated with @tool or @petaltool).

        Args:
            name (str): Name to register the tool under.
            fn (BaseTool): The tool (must be a BaseTool instance).

        Returns:
            ToolFactory: self (for chaining)
        """
        if not isinstance(fn, BaseTool):
            raise TypeError(
                f"Tool '{name}' must be decorated with @tool or @petaltool and be a BaseTool instance. "
                f"Got type: {type(fn)}."
            )
        self._registry[name] = fn
        return self

    def resolve(self, name: str) -> BaseTool:
        """
        Retrieve a tool by name. Only BaseTool instances are supported.

        Args:
            name (str): The name of the tool.

        Returns:
            BaseTool: The registered tool.

        Raises:
            KeyError: If the tool is not found.
            TypeError: If the registered object is not a BaseTool (should not happen).
        """
        # Check if this is an MCP tool that's still loading
        if name.startswith("mcp:") and name not in self._registry:
            parts = name.split(":")
            if len(parts) >= 3:
                server_name = parts[1]
                namespace = f"mcp:{server_name}"
                event = self._mcp_loaded.get(namespace)
                if event:
                    asyncio.create_task(self._wait_for_mcp_and_resolve(name, event))
                    raise KeyError(
                        f"MCP tool '{name}' is still loading. Please wait for it to complete."
                    )
                else:
                    raise KeyError(
                        f"MCP tool '{name}' not found. Server '{server_name}' has not been registered."
                    )

        if name not in self._registry:
            raise KeyError(f"Tool '{name}' not found in registry.")

        tool = self._registry[name]
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"Tool '{name}' is not a BaseTool instance. All tools must be registered with @tool or @petaltool."
            )
        return tool

    async def _wait_for_mcp_and_resolve(self, name: str, event: asyncio.Event) -> None:
        """
        Wait for MCP tools to load and then resolve the specific tool.
        """
        await event.wait()
        if name not in self._registry:
            raise KeyError(f"Tool '{name}' not found after MCP loading completed.")

    def list(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List[str]: Sorted list of tool names.
        """
        return sorted(self._registry.keys())

    def add_mcp(
        self,
        server_name: str,
        resolver: Callable[[], Coroutine[Any, Any, List[Any]]] | None = None,
        mcp_config: dict[str, dict[str, object]] | None = None,
    ) -> "ToolFactory":
        """
        Asynchronously resolve and register MCP tools under the "mcp" namespace.
        This method launches the resolver as a background task, so the framework is not blocked.
        Once resolved, each tool is registered as f"mcp:{server_name}:{tool.name}".

        Args:
            server_name (str): Name of the MCP server (typically the config key) to avoid naming collisions.
            resolver (Callable[[], Coroutine] | None): Async function returning a list of tool objects (must have .name and be callable).
                If None, uses default resolver with mcp_config.
            mcp_config (dict | None): MCP server configuration for default resolver. Required if resolver is None.

        Returns:
            ToolFactory: self (for chaining)
        """
        # Validate parameters immediately
        if resolver is None and mcp_config is None:
            raise ValueError("mcp_config is required when resolver is None")

        namespace = f"mcp:{server_name}"
        event = asyncio.Event()
        self._mcp_loaded[namespace] = event

        async def _load_and_register() -> None:
            if resolver is not None:
                tools = await resolver()
            else:
                from langchain_mcp_adapters.client import MultiServerMCPClient

                client = MultiServerMCPClient(mcp_config)
                tools = await client.get_tools()

            for tool in tools:
                tool_name = f"{namespace}:{getattr(tool, 'name', tool.name)}"
                self.add(tool_name, tool)
            event.set()

        task = asyncio.create_task(_load_and_register())
        self._mcp_tasks[namespace] = task
        return self

    async def await_mcp_loaded(self, server_name: str) -> None:
        """
        Await until MCP tools for a given server are loaded.
        """
        namespace = f"mcp:{server_name}"
        event = self._mcp_loaded.get(namespace)
        if event:
            await event.wait()
