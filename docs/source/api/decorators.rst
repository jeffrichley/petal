Decorators
==========

Petal provides several decorators for tool registration, MCP integration, and automatic discovery.

.. automodule:: petal.core.decorators
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The decorators module provides convenient ways to register tools and MCP servers with automatic namespace management and discovery integration.

Key Features
------------

- **Automatic Namespacing**: Tools are automatically namespaced with their module name
- **MCP Integration**: Decorators for MCP server and tool registration
- **LangChain Compatibility**: Tools are compatible with LangChain's tool system
- **Discovery Integration**: Decorated tools are automatically discovered
- **Type Safety**: Full type hints and validation support

@petaltool
----------

The main decorator for registering tools with automatic namespace management.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from petal.core.decorators import petaltool

    @petaltool
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    # Tool is automatically registered as "module_name:add_numbers"
    # Can be resolved as "add_numbers" (base name) or full namespaced name

With Custom Name
~~~~~~~~~~~~~~~

.. code-block:: python

    @petaltool("calculator:add")
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    # Tool is registered as "calculator:add"

With Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @petaltool(
        "calculator:multiply",
        description="Multiply two numbers",
        return_direct=False,
        infer_schema=True
    )
    def multiply_numbers(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pydantic import BaseModel

    class MathInput(BaseModel):
        a: float
        b: float

    @petaltool(
        "calculator:complex_operation",
        args_schema=MathInput,
        infer_schema=False,
        parse_docstring=True
    )
    def complex_math_operation(input_data: MathInput) -> float:
        """Perform a complex mathematical operation."""
        return input_data.a ** 2 + input_data.b ** 2

Async Functions
~~~~~~~~~~~~~~~

.. code-block:: python

    @petaltool("api:fetch_data")
    async def fetch_data(url: str) -> dict:
        """Fetch data from a URL."""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()

Namespace Management
-------------------

The `@petaltool` decorator automatically manages namespaces:

**Automatic Namespacing**
    Tools are automatically namespaced with their module name:
    .. code-block:: python

        # In module "my_tools.math"
        @petaltool
        def add(a: float, b: float) -> float:
            return a + b
        # Registered as "my_tools.math:add"

**Custom Namespacing**
    You can provide custom namespaces:
    .. code-block:: python

        @petaltool("calculator:add")
        def add(a: float, b: float) -> float:
            return a + b
        # Registered as "calculator:add"

**Main Module Handling**
    Tools in `__main__` module use just the function name:
    .. code-block:: python

        # In __main__ (script execution)
        @petaltool
        def add(a: float, b: float) -> float:
            return a + b
        # Registered as "add"

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

@petalmcp
---------

Decorator for creating MCP server classes.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from petal.core.decorators import petalmcp

    @petalmcp("filesystem", {"command": "mcp-server-filesystem"})
    class FileSystemServer:
        """MCP server for filesystem operations."""
        pass

    # Server is registered with ToolFactory.add_mcp()

With Complex Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @petalmcp("database", {
        "command": "mcp-server-database",
        "args": ["--host", "localhost", "--port", "5432"],
        "env": {"DB_PASSWORD": "secret"}
    })
    class DatabaseServer:
        """MCP server for database operations."""
        pass

@petalmcp_tool
--------------

Decorator for creating MCP tool functions.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from petal.core.decorators import petalmcp_tool

    @petalmcp_tool("filesystem:read_file")
    def read_file(path: str) -> str:
        """Read a file from the filesystem."""
        # Implementation would use MCP client
        pass

    # Function is registered as "mcp:filesystem:read_file"

With Complex Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @petalmcp_tool("database:query")
    def execute_query(query: str, parameters: dict = None) -> list:
        """Execute a database query."""
        # Implementation would use MCP client
        pass

MCP Integration
---------------

MCP tools are automatically loaded and registered:

**Background Loading**
    MCP tools are loaded in the background to avoid blocking:
    .. code-block:: python

        # Tools are loaded asynchronously
        factory = ToolFactory()
        factory.add_mcp("filesystem", mcp_config={"command": "mcp-server-filesystem"})

**Loading States**
    You can wait for MCP tools to load:
    .. code-block:: python

        await factory.await_mcp_loaded("filesystem")
        tool = factory.resolve("mcp:filesystem:read_file")

**Error Handling**
    MCP loading errors are handled gracefully:
    .. code-block:: python

        try:
            tool = factory.resolve("mcp:filesystem:read_file")
        except KeyError as e:
            print(f"MCP tool not loaded: {e}")

Utility Functions
-----------------

auto_namespace()
~~~~~~~~~~~~~~~

Generate an auto-namespaced name for a function.

.. code-block:: python

    from petal.core.decorators import auto_namespace

    def my_function():
        pass

    name = auto_namespace(my_function)
    # Returns "module_name:my_function" or just "my_function" for __main__

Examples
--------

**Complete Tool Registration Example**
    .. code-block:: python

        from petal.core.decorators import petaltool
        from petal.core.registry import ToolRegistry

        @petaltool("math:add")
        def add(a: float, b: float) -> float:
            """Add two numbers together."""
            return a + b

        @petaltool("math:multiply")
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers together."""
            return a * b

        # Tools are automatically registered
        registry = ToolRegistry()

        # Can be resolved by full name
        add_tool = await registry.resolve("math:add")

        # Or by base name (if unambiguous)
        multiply_tool = await registry.resolve("multiply")

**MCP Integration Example**
    .. code-block:: python

        from petal.core.decorators import petalmcp, petalmcp_tool
        from petal.core.tool_factory import ToolFactory

        @petalmcp("filesystem", {"command": "mcp-server-filesystem"})
        class FileSystemServer:
            pass

        @petalmcp_tool("filesystem:read_file")
        def read_file(path: str) -> str:
            """Read a file from the filesystem."""
            pass

        @petalmcp_tool("filesystem:write_file")
        def write_file(path: str, content: str) -> bool:
            """Write content to a file."""
            pass

        # MCP tools are automatically registered
        factory = ToolFactory()

        # Wait for tools to load
        await factory.await_mcp_loaded("filesystem")

        # Use the tools
        read_tool = factory.resolve("mcp:filesystem:read_file")
        write_tool = factory.resolve("mcp:filesystem:write_file")

**Complex Tool with Schema**
    .. code-block:: python

        from petal.core.decorators import petaltool
        from pydantic import BaseModel
        from typing import List

        class SearchInput(BaseModel):
            query: str
            filters: List[str] = []
            limit: int = 10

        class SearchResult(BaseModel):
            items: List[dict]
            total: int
            query: str

        @petaltool(
            "search:web_search",
            args_schema=SearchInput,
            infer_schema=False,
            parse_docstring=True
        )
        def web_search(input_data: SearchInput) -> SearchResult:
            """
            Search the web for information.

            Args:
                input_data: Search parameters including query and filters

            Returns:
                SearchResult with items and metadata
            """
            # Implementation would perform web search
            return SearchResult(
                items=[{"title": "Result", "url": "http://example.com"}],
                total=1,
                query=input_data.query
            )

**Async Tool with Error Handling**
    .. code-block:: python

        from petal.core.decorators import petaltool
        import aiohttp
        from typing import Optional

        @petaltool("api:fetch_weather")
        async def fetch_weather(city: str, country: Optional[str] = None) -> dict:
            """
            Fetch weather information for a city.

            Args:
                city: The city name
                country: Optional country code

            Returns:
                Weather data dictionary
            """
            try:
                url = f"https://api.weatherapi.com/v1/current.json?key=YOUR_KEY&q={city}"
                if country:
                    url += f",{country}"

                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            return {"error": f"HTTP {response.status}"}
            except Exception as e:
                return {"error": str(e)}

Error Handling
--------------

The decorators provide comprehensive error handling:

**Type Validation**
    .. code-block:: python

        @petaltool("test:invalid")
        def invalid_tool():
            # Missing type hints - will raise validation error
            pass

**Duplicate Registration**
    .. code-block:: python

        @petaltool("test:duplicate")
        def tool1():
            pass

        @petaltool("test:duplicate")  # Will overwrite previous registration
        def tool2():
            pass

**MCP Configuration Errors**
    .. code-block:: python

        # Missing configuration
        @petalmcp("test")  # Will raise ValueError
        class TestServer:
            pass

Best Practices
--------------

**Naming Conventions**
    - Use descriptive, hierarchical names: `module:category:function`
    - Keep base names unique within your project
    - Use consistent naming patterns

**Documentation**
    - Always provide docstrings for tools
    - Use clear parameter descriptions
    - Include return type information

**Error Handling**
    - Handle errors gracefully in tool implementations
    - Return meaningful error messages
    - Use appropriate HTTP status codes for API tools

**Type Safety**
    - Use type hints for all parameters and return values
    - Use Pydantic models for complex inputs
    - Validate inputs when necessary

**MCP Integration**
    - Use descriptive server names
    - Provide complete MCP configurations
    - Handle MCP loading states appropriately

**Performance**
    - Keep tool implementations efficient
    - Use async functions for I/O operations
    - Cache expensive operations when appropriate
