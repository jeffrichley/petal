ToolFactory
===========

Async-friendly registry for callable tools with MCP support and background loading.

.. automodule:: petal.core.tool_factory
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ToolFactory provides an async-friendly registry for callable tools with support for MCP (Model Context Protocol) integration. It handles both sync and async tools, provides background loading for MCP tools, and offers a fluent interface for configuration.

Key Features
------------

- **Async Support**: Handles both sync and async tools seamlessly
- **MCP Integration**: Background loading of MCP tools with loading state management
- **Tool Resolution**: Resolves tools by name with proper error handling
- **Loading States**: Manages MCP tool loading states with async events
- **Chaining Interface**: Fluent interface for configuration
- **Thread Safety**: Safe for concurrent access
- **Error Handling**: Comprehensive error messages for missing or loading tools

Basic Usage
-----------

.. code-block:: python

    from petal.core.tool_factory import ToolFactory
    from langchain_core.tools import tool

    # Create tool factory
    factory = ToolFactory()

    # Register a simple tool
    @tool("calculator:add")
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    factory.add("calculator:add", add)

    # Resolve and use the tool
    tool = factory.resolve("calculator:add")
    result = await tool.ainvoke({"a": 5, "b": 3})
    print(result)  # 8

MCP Integration
---------------

The ToolFactory provides seamless MCP integration with background loading:

.. code-block:: python

    from petal.core.tool_factory import ToolFactory

    factory = ToolFactory()

    # Add MCP tools with custom resolver
    async def custom_resolver():
        # Your custom MCP tool resolution logic
        return [tool1, tool2, tool3]

    factory.add_mcp("filesystem", custom_resolver)

    # Or use default resolver with MCP config
    mcp_config = {
        "filesystem": {
            "command": "mcp-server-filesystem",
            "args": ["--config", "config.json"]
        }
    }
    factory.add_mcp("filesystem", mcp_config=mcp_config)

    # Wait for MCP tools to load
    await factory.await_mcp_loaded("filesystem")

    # Use MCP tools
    tool = factory.resolve("mcp:filesystem:read_file")
    result = await tool.ainvoke({"path": "/etc/hosts"})

MCP Tool Loading States
-----------------------

The factory manages MCP tool loading states:

.. code-block:: python

    from petal.core.tool_factory import ToolFactory

    factory = ToolFactory()

    # Register MCP server (starts background loading)
    factory.add_mcp("filesystem", mcp_config=config)

    # Try to resolve before loading completes
    try:
        tool = factory.resolve("mcp:filesystem:read_file")
    except KeyError as e:
        print(f"Tool still loading: {e}")

    # Wait for loading to complete
    await factory.await_mcp_loaded("filesystem")

    # Now resolve successfully
    tool = factory.resolve("mcp:filesystem:read_file")

Method Reference
----------------

.. method:: ToolFactory.__init__() -> None

    Initialize the tool registry.

.. method:: ToolFactory.add(name: str, fn: BaseTool) -> ToolFactory

    Register a tool by name. Must be a BaseTool instance.

    Args:
        name: Name to register the tool under
        fn: The tool (must be a BaseTool instance)

    Returns:
        self for chaining

    Raises:
        TypeError: If fn is not a BaseTool instance

    Example:
        .. code-block:: python

            @tool("math:add")
            def add(a: int, b: int) -> int:
                return a + b

            factory.add("math:add", add)

.. method:: ToolFactory.resolve(name: str) -> BaseTool

    Retrieve a tool by name. Only BaseTool instances are supported.

    Args:
        name: The name of the tool

    Returns:
        The registered tool

    Raises:
        KeyError: If the tool is not found or still loading
        TypeError: If the registered object is not a BaseTool

    Example:
        .. code-block:: python

            tool = factory.resolve("math:add")
            result = await tool.ainvoke({"a": 2, "b": 3})

.. method:: ToolFactory.list() -> List[str]

    List all registered tool names.

    Returns:
        Sorted list of tool names

    Example:
        .. code-block:: python

            tools = factory.list()
            print(tools)  # ['math:add', 'mcp:filesystem:read_file']

.. method:: ToolFactory.add_mcp(server_name: str, resolver: Optional[Callable] = None, mcp_config: Optional[Dict] = None) -> ToolFactory

    Asynchronously resolve and register MCP tools under the "mcp" namespace.

    Args:
        server_name: Name of the MCP server to avoid naming collisions
        resolver: Async function returning a list of tool objects (optional)
        mcp_config: MCP server configuration for default resolver (required if resolver is None)

    Returns:
        self for chaining

    Raises:
        ValueError: If mcp_config is required but not provided

    Example:
        .. code-block:: python

            # With custom resolver
            async def my_resolver():
                return [tool1, tool2]

            factory.add_mcp("my_server", resolver=my_resolver)

            # With default resolver
            config = {"server": {"command": "mcp-server-filesystem"}}
            factory.add_mcp("filesystem", mcp_config=config)

.. method:: ToolFactory.await_mcp_loaded(server_name: str) -> None

    Await until MCP tools for a given server are loaded.

    Args:
        server_name: Name of the MCP server

    Example:
        .. code-block:: python

            factory.add_mcp("filesystem", mcp_config=config)
            await factory.await_mcp_loaded("filesystem")
            # Now safe to resolve tools

Error Handling
--------------

The ToolFactory provides comprehensive error handling:

**Missing Tools**
    .. code-block:: python

        try:
            tool = factory.resolve("nonexistent")
        except KeyError as e:
            print(f"Tool not found: {e}")

**MCP Tools Not Registered**
    .. code-block:: python

        try:
            tool = factory.resolve("mcp:unregistered:tool")
        except KeyError as e:
            print(f"MCP server not registered: {e}")

**MCP Tools Still Loading**
    .. code-block:: python

        factory.add_mcp("filesystem", mcp_config=config)
        try:
            tool = factory.resolve("mcp:filesystem:read_file")
        except KeyError as e:
            print(f"Tool still loading: {e}")
            await factory.await_mcp_loaded("filesystem")

**Invalid Tool Types**
    .. code-block:: python

        try:
            factory.add("invalid", "not a tool")
        except TypeError as e:
            print(f"Must be BaseTool instance: {e}")

**Missing MCP Configuration**
    .. code-block:: python

        try:
            factory.add_mcp("server")  # No resolver, no config
        except ValueError as e:
            print(f"Configuration required: {e}")

Integration with AgentFactory
-----------------------------

The ToolFactory integrates seamlessly with AgentFactory:

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState
    from petal.core.tool_factory import ToolFactory

    # Create and configure tool factory
    tool_factory = ToolFactory()
    tool_factory.add_mcp("filesystem", mcp_config=config)

    # Create agent with tool discovery
    agent = (
        AgentFactory(DefaultState)
        .with_chat(prompt_template="Use tools to help the user")
        .with_tools(["mcp:filesystem:read_file", "calculator:add"])
        .with_tool_discovery(enabled=True)
        .build()
    )

    # Wait for MCP tools to load
    await tool_factory.await_mcp_loaded("filesystem")

    # Run agent
    result = await agent.arun({"messages": []})

Performance Considerations
-------------------------

- **Background Loading**: MCP tools load in the background to avoid blocking
- **Lazy Resolution**: Tools are only resolved when needed
- **Caching**: Resolved tools are cached for performance
- **Async Events**: Efficient async event handling for loading states

Thread Safety
-------------

The ToolFactory is designed for concurrent access:

- **Async Operations**: All operations are async-friendly
- **Event-Based Loading**: MCP loading uses async events
- **Immutable State**: Tool registration is thread-safe
- **Concurrent Resolution**: Multiple threads can resolve tools simultaneously
