Tool Registry
============

Singleton tool registry with lazy discovery capabilities and thread safety.

.. automodule:: petal.core.registry
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ToolRegistry is a thread-safe singleton that provides tool registration, discovery, and resolution capabilities. It supports multiple discovery strategies and handles tool name ambiguity gracefully.

Key Features
------------

- **Singleton Pattern**: Single registry across all agents and processes
- **Thread Safety**: Thread-safe operations with proper locking
- **Lazy Discovery**: Tools are discovered only when needed
- **Multiple Strategies**: Decorator, config, folder, and MCP discovery
- **Namespace Support**: Organized tool naming with `namespace:tool` format
- **Caching**: Performance optimization with discovery caching
- **Ambiguity Resolution**: Handles ambiguous tool names gracefully

Basic Usage
-----------

.. code-block:: python

    from petal.core.registry import ToolRegistry
    from petal.core.decorators import petaltool

    @petaltool("math:add")
    def add(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    # Get the singleton registry
    registry = ToolRegistry()

    # Resolve tool by name
    tool = await registry.resolve("math:add")

    # List all registered tools
    tools = registry.list()
    print(tools)  # ['math:add']

Discovery Strategies
-------------------

The registry supports multiple discovery strategies:

**DecoratorDiscovery**
    Discovers tools decorated with `@petaltool`:
    .. code-block:: python

        from petal.core.discovery import DecoratorDiscovery

        registry = ToolRegistry()
        registry.add_discovery_strategy(DecoratorDiscovery())

**ConfigDiscovery**
    Discovers tools from YAML configuration files:
    .. code-block:: python

        from petal.core.discovery import ConfigDiscovery

        registry = ToolRegistry()
        registry.add_discovery_strategy(ConfigDiscovery(["config/tools.yaml"]))

**FolderDiscovery**
    Scans directories for Python files containing tools:
    .. code-block:: python

        from petal.core.discovery import FolderDiscovery

        registry = ToolRegistry()
        registry.add_discovery_strategy(FolderDiscovery(["tools/"]))

**Multiple Strategies**
    Combine multiple discovery strategies:
    .. code-block:: python

        from petal.core.discovery import DecoratorDiscovery, ConfigDiscovery, FolderDiscovery

        registry = ToolRegistry()
        registry.add_discovery_strategy(DecoratorDiscovery())
        registry.add_discovery_strategy(ConfigDiscovery(["config/tools.yaml"]))
        registry.add_discovery_strategy(FolderDiscovery(["tools/"]))

Tool Resolution
---------------

Tools can be resolved in several ways:

**Full Namespace**
    .. code-block:: python

        tool = await registry.resolve("my_module:add_numbers")

**Base Name (Single Match)**
    .. code-block:: python

        tool = await registry.resolve("add_numbers")  # If only one match

**Base Name (Multiple Matches)**
    .. code-block:: python

        # Raises AmbiguousToolNameError if multiple tools have same base name
        try:
            tool = await registry.resolve("add")
        except AmbiguousToolNameError as e:
            print(f"Multiple tools found: {e.matching_tools}")

**Direct Registration**
    .. code-block:: python

        from langchain.tools import tool

        @tool("calculator:add")
        def add(a: float, b: float) -> float:
            return a + b

        registry = ToolRegistry()
        registry.add("calculator:add", add)

Method Reference
----------------

.. method:: ToolRegistry.__new__() -> ToolRegistry

    Ensure singleton pattern with thread safety.

    Returns:
        The singleton ToolRegistry instance

.. method:: ToolRegistry.__init__() -> None

    Initialize the registry (only called once due to singleton).

.. method:: ToolRegistry.add(name: str, tool: BaseTool) -> ToolRegistry

    Register a tool by name. Must be a BaseTool instance.

    Args:
        name: The name to register the tool under
        tool: The tool (must be a BaseTool instance)

    Returns:
        self for chaining

    Raises:
        TypeError: If tool is not a BaseTool instance

.. method:: ToolRegistry.resolve(name: str) -> BaseTool

    Resolve tool with lazy discovery if not found.

    Args:
        name: The name of the tool to resolve

    Returns:
        The resolved BaseTool instance

    Raises:
        KeyError: If the tool is not found
        AmbiguousToolNameError: If multiple tools match the base name

.. method:: ToolRegistry.list() -> List[str]

    List all registered tool names.

    Returns:
        Sorted list of tool names

.. method:: ToolRegistry.add_discovery_strategy(strategy: DiscoveryStrategy) -> ToolRegistry

    Add a discovery strategy to the chain.

    Args:
        strategy: The discovery strategy to add

    Returns:
        self for chaining

.. method:: ToolRegistry._reset_for_testing() -> None

    Reset the registry for testing purposes.

Discovery Strategy Interface
----------------------------

.. class:: DiscoveryStrategy

    Abstract base for tool discovery strategies.

    .. method:: discover(name: str) -> Optional[BaseTool]

        Discover a tool by name. Returns None if not found.

        Args:
            name: The name of the tool to discover

        Returns:
            The discovered tool or None if not found

Error Handling
--------------

The registry provides comprehensive error handling:

**AmbiguousToolNameError**
    Raised when multiple tools have the same base name:
    .. code-block:: python

        try:
            tool = await registry.resolve("add")
        except AmbiguousToolNameError as e:
            print(f"Ambiguous tool name '{e.base_name}'")
            print(f"Multiple tools found: {e.matching_tools}")
            # Use full namespaced name to disambiguate
            tool = await registry.resolve("math:add")

**KeyError**
    Raised when a tool is not found:
    .. code-block:: python

        try:
            tool = await registry.resolve("nonexistent_tool")
        except KeyError as e:
            print(f"Tool not found: {e}")

**TypeError**
    Raised when trying to register non-BaseTool objects:
    .. code-block:: python

        try:
            registry.add("invalid", "not a tool")
        except TypeError as e:
            print(f"Invalid tool type: {e}")

Thread Safety
-------------

The registry is designed to be thread-safe:

**Singleton Initialization**
    Thread-safe singleton pattern with double-checked locking:
    .. code-block:: python

        # Multiple threads get the same instance
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()
        assert registry1 is registry2  # Same instance

**Concurrent Access**
    Safe concurrent access to registry methods:
    .. code-block:: python

        import asyncio
        import threading

        async def worker():
            registry = ToolRegistry()
            tool = await registry.resolve("math:add")
            return tool

        # Multiple threads can safely access the registry
        tasks = [worker() for _ in range(10)]
        results = await asyncio.gather(*tasks)

Caching
-------

The registry uses caching for performance optimization:

**Discovery Cache**
    Failed discoveries are cached to avoid repeated lookups:
    .. code-block:: python

        # First attempt - tries discovery
        try:
            tool = await registry.resolve("nonexistent_tool")
        except KeyError:
            pass

        # Second attempt - uses cached result (faster)
        try:
            tool = await registry.resolve("nonexistent_tool")
        except KeyError:
            pass

**Tool Cache**
    Successfully discovered tools are cached in the registry:
    .. code-block:: python

        # First resolution - discovers and caches
        tool1 = await registry.resolve("math:add")

        # Second resolution - uses cached tool
        tool2 = await registry.resolve("math:add")

        assert tool1 is tool2  # Same cached instance

Examples
--------

**Basic Tool Registration**
    .. code-block:: python

        from petal.core.registry import ToolRegistry
        from petal.core.decorators import petaltool

        @petaltool("math:add")
        def add(a: float, b: float) -> float:
            """Add two numbers together."""
            return a + b

        @petaltool("math:multiply")
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers together."""
            return a * b

        registry = ToolRegistry()

        # Tools are automatically registered by decorators
        add_tool = await registry.resolve("math:add")
        multiply_tool = await registry.resolve("math:multiply")

        # List all tools
        all_tools = registry.list()
        print(all_tools)  # ['math:add', 'math:multiply']

**Discovery Strategy Chain**
    .. code-block:: python

        from petal.core.registry import ToolRegistry
        from petal.core.discovery import DecoratorDiscovery, ConfigDiscovery, FolderDiscovery

        registry = ToolRegistry()

        # Add discovery strategies in order of preference
        registry.add_discovery_strategy(DecoratorDiscovery())
        registry.add_discovery_strategy(ConfigDiscovery(["config/tools.yaml"]))
        registry.add_discovery_strategy(FolderDiscovery(["tools/"]))

        # Try to resolve a tool - will try each strategy in order
        try:
            tool = await registry.resolve("web_search")
        except KeyError:
            print("Tool not found in any discovery strategy")

**Ambiguous Tool Resolution**
    .. code-block:: python

        from petal.core.registry import ToolRegistry, AmbiguousToolNameError
        from petal.core.decorators import petaltool

        @petaltool("math:add")
        def add(a: float, b: float) -> float:
            return a + b

        @petaltool("calculator:add")
        def calculator_add(a: float, b: float) -> float:
            return a + b

        registry = ToolRegistry()

        # Try to resolve by base name - will fail due to ambiguity
        try:
            tool = await registry.resolve("add")
        except AmbiguousToolNameError as e:
            print(f"Ambiguous tool name: {e.base_name}")
            print(f"Multiple tools found: {e.matching_tools}")
            # Use full namespaced name to disambiguate
            tool = await registry.resolve("math:add")

**Direct Tool Registration**
    .. code-block:: python

        from petal.core.registry import ToolRegistry
        from langchain.tools import tool

        @tool("custom:greet")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        registry = ToolRegistry()
        registry.add("custom:greet", greet)

        tool = await registry.resolve("custom:greet")
        result = tool.invoke({"name": "Alice"})
        print(result)  # "Hello, Alice!"

**Integration with AgentFactory**
    .. code-block:: python

        from petal.core.factory import AgentFactory, DefaultState
        from petal.core.registry import ToolRegistry
        from petal.core.decorators import petaltool

        @petaltool("calculator:add")
        def add(a: float, b: float) -> float:
            return a + b

        # AgentFactory uses the singleton registry internally
        agent = (
            AgentFactory(DefaultState)
            .with_chat(
                prompt_template="Calculate {expression} for me.",
                system_prompt="You are a helpful math assistant."
            )
            .with_tools(["calculator:add"])  # Resolved via registry
            .build()
        )

        result = await agent.arun({
            "name": "User",
            "expression": "5 + 3",
            "messages": []
        })

**Testing with Registry Reset**
    .. code-block:: python

        import pytest
        from petal.core.registry import ToolRegistry

        @pytest.fixture
        def clean_registry():
            registry = ToolRegistry()
            registry._reset_for_testing()
            yield registry
            registry._reset_for_testing()

        def test_tool_registration(clean_registry):
            # Each test starts with a clean registry
            assert clean_registry.list() == []

            # Add test tools
            clean_registry.add("test:tool", some_tool)
            assert "test:tool" in clean_registry.list()

Performance Considerations
-------------------------

**Lazy Discovery**
    Tools are only discovered when needed, improving startup time:
    .. code-block:: python

        # Discovery strategies are not executed until resolve() is called
        registry = ToolRegistry()
        registry.add_discovery_strategy(DecoratorDiscovery())  # No scanning yet

        # Scanning happens only when resolving
        tool = await registry.resolve("math:add")  # Discovery executed here

**Caching Benefits**
    Repeated lookups use cached results:
    .. code-block:: python

        import time

        # First lookup - includes discovery time
        start = time.time()
        tool1 = await registry.resolve("math:add")
        first_lookup_time = time.time() - start

        # Second lookup - uses cache
        start = time.time()
        tool2 = await registry.resolve("math:add")
        second_lookup_time = time.time() - start

        assert second_lookup_time < first_lookup_time

**Thread Safety Overhead**
    Minimal overhead for thread safety:
    .. code-block:: python

        import asyncio
        import time

        async def concurrent_resolve():
            registry = ToolRegistry()
            start = time.time()
            tools = await asyncio.gather(*[
                registry.resolve("math:add") for _ in range(100)
            ])
            return time.time() - start

        # Thread-safe operations have minimal overhead
        total_time = await concurrent_resolve()
        print(f"100 concurrent resolves took: {total_time:.3f}s")

Best Practices
--------------

**Tool Naming**
    - Use descriptive, hierarchical names: `module:category:function`
    - Keep base names unique within your project
    - Use consistent naming patterns

**Discovery Strategy Order**
    - Order strategies by preference (fastest/most reliable first)
    - Use DecoratorDiscovery for development tools
    - Use ConfigDiscovery for production tools
    - Use FolderDiscovery for legacy tools

**Error Handling**
    - Always handle AmbiguousToolNameError for base name resolution
    - Provide fallback mechanisms for missing tools
    - Log discovery failures for debugging

**Performance**
    - Use lazy discovery to improve startup time
    - Leverage caching for frequently accessed tools
    - Consider tool preloading for critical paths

**Testing**
    - Use `_reset_for_testing()` to isolate tests
    - Mock discovery strategies for unit tests
    - Test both success and failure scenarios
