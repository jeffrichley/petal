Architecture Overview
=====================

This document provides an overview of the Petal framework architecture and design principles.

Design Philosophy
-----------------

Petal is built around several core principles:

1. **Pythonic API**: Fluent, chainable interfaces that feel natural to Python developers
2. **Zero-Config Discovery**: Automatic tool and agent discovery with sensible defaults
3. **LangGraph Compatibility**: All components can be used as LangGraph nodes
4. **Type Safety**: Comprehensive type hints and Pydantic validation throughout
5. **Extensibility**: Plugin system for custom step types and strategies
6. **Thread Safety**: Singleton patterns and thread-safe operations
7. **Performance**: Lazy loading, caching, and efficient resource management
8. **Async-First**: All operations are async-friendly for better performance

Core Components
---------------

AgentFactory (High-level API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main entry point for creating agents. Provides a fluent interface that uses AgentBuilder internally:

- **Steps**: Individual functions that process state
- **Prompts**: Template strings for LLM interactions
- **System Prompts**: Dynamic system prompts with state variable interpolation
- **Configuration**: LLM settings, memory, logging, etc.
- **Tool Integration**: Automatic tool discovery and binding
- **ReAct Loops**: Reasoning and tool use workflows
- **YAML Configuration**: Declarative agent definitions
- **Structured Output**: Pydantic model binding for LLM responses
- **Tool Discovery**: Configurable discovery strategies

.. code-block:: python

   agent = (AgentFactory(DefaultState)
       .add(my_function)
       .with_chat(
           prompt_template="Process {input}",
           system_prompt="You are a helpful assistant."
       )
       .with_tools(["calculator:add"])
       .with_structured_output(MyModel)
       .with_tool_discovery(enabled=True, folders=["tools/"])
       .build())

AgentBuilder (Lower-level API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provides explicit control over agent building using composition patterns:

- **Step Configuration**: Explicit step type and configuration
- **Named Parameters**: Type-safe LLM configuration with named parameters
- **Configuration Objects**: Direct access to Pydantic configuration models
- **Step Registry**: Pluggable step strategies
- **Structured Output**: Pydantic model binding for LLM responses
- **Memory Configuration**: Configurable memory backends
- **Logging Configuration**: Comprehensive logging options
- **Graph Configuration**: LangGraph topology and edge configuration

.. code-block:: python

   builder = AgentBuilder(MyState)
   agent = (
       builder.with_step("custom", step_function=my_function)
       .with_step("llm", prompt_template="Process {input}")
       .with_system_prompt("You are a helpful assistant.")
       .with_llm(provider="openai", model="gpt-4o-mini")
       .with_structured_output(MyModel)
       .with_memory({"memory_type": "conversation"})
       .with_logging({"level": "INFO", "include_state": True})
       .build()
   )

Tool Registry (Singleton)
~~~~~~~~~~~~~~~~~~~~~~~~~

Thread-safe singleton tool registry with lazy discovery:

- **Singleton Pattern**: Single registry across all agents
- **Lazy Discovery**: Tools are discovered only when needed
- **Multiple Strategies**: Decorator, config, folder, and MCP discovery
- **Namespace Support**: Organized tool naming with `namespace:tool` format
- **Caching**: Performance optimization with discovery caching
- **Thread Safety**: Thread-safe operations with proper locking
- **Ambiguity Resolution**: Handles ambiguous tool names gracefully

.. code-block:: python

   from petal.core.registry import ToolRegistry
   from petal.core.decorators import petaltool

   @petaltool("math:add")
   def add(a: float, b: float) -> float:
       return a + b

   registry = ToolRegistry()  # Singleton
   tool = await registry.resolve("math:add")

Tool Factory
~~~~~~~~~~~~

Async-friendly registry for callable tools with MCP support:

- **Async Support**: Handles both sync and async tools
- **MCP Integration**: Background loading of MCP tools
- **Tool Resolution**: Resolves tools by name with proper error handling
- **Loading States**: Manages MCP tool loading states
- **Chaining Interface**: Fluent interface for configuration

.. code-block:: python

   from petal.core.tool_factory import ToolFactory

   factory = ToolFactory()
   factory.add_mcp("filesystem", mcp_config={"command": "mcp-server-filesystem"})
   await factory.await_mcp_loaded("filesystem")
   tool = factory.resolve("mcp:filesystem:read_file")

Step Strategy System
--------------------

The framework uses the Strategy pattern for step management:

**StepStrategy ABC**
   Abstract base class defining the interface for step creation strategies.

**LLMStepStrategy**
   Strategy for creating LLM steps with prompt templates, system prompts, and tool binding.

**CustomStepStrategy**
   Strategy for creating custom function steps (sync or async).

**ToolStepStrategy**
   Strategy for creating tool execution steps with scratchpad support.

**ReactStepStrategy**
   Strategy for creating ReAct reasoning loops with tool integration.

**StepRegistry**
   Registry for managing and discovering step strategies with thread safety.

.. code-block:: python

   # Register a custom step strategy
   registry = StepRegistry()
   registry.register("my_step", MyStepStrategy)

   # Use the strategy
   strategy = registry.get_strategy("my_step")
   step = await strategy.create_step(config)

Configuration Management
-----------------------

Petal uses Pydantic models for configuration management:

**AgentConfig**
   Main configuration object containing all agent settings.

**StepConfig**
   Configuration for individual steps with strategy type and parameters.

**LLMConfig**
   LLM-specific configuration with named parameters and validation.

**MemoryConfig**
   Memory backend configuration with type validation.

**GraphConfig**
   LangGraph configuration options including topology and retry settings.

**LoggingConfig**
   Logging configuration with level and state inclusion options.

.. code-block:: python

   config = AgentConfig(
       state_type=MyState,
       steps=[StepConfig(strategy_type="llm", config={...})],
       llm_config=LLMConfig(provider="openai", model="gpt-4o-mini"),
       memory=MemoryConfig(memory_type="conversation"),
       graph_config=GraphConfig(graph_type="linear"),
       logging_config=LoggingConfig(level="INFO")
   )

State Management
----------------

Petal uses TypedDict for strongly-typed state:

- **Input state**: Initial data passed to the agent
- **Step state**: Data processed by each step
- **Output state**: Final result from the agent
- **Message state**: Automatic message handling with `add_messages`

State flows through the system and can be modified by:

- **Steps**: Functions that receive and return state
- **LLM**: Can access and modify state through prompts
- **System prompts**: Support state variable interpolation
- **Tools**: Can read and write state during execution

.. code-block:: python

   class MyState(TypedDict):
       messages: Annotated[list, add_messages]
       name: str
       personality: str

   async def custom_step(state: dict) -> dict:
       state["personality"] = "pirate"
       return state

   # State variables can be used in prompts
   agent.with_chat(
       prompt_template="Hello {name}!",
       system_prompt="You are a {personality} assistant."
   )

StateTypeFactory
---------------

Handles dynamic state type creation:

- **Message support**: Automatically adds message support to state types
- **Caching**: Caches created types for performance
- **Type safety**: Ensures proper TypedDict structure
- **Mergeable fields**: Supports automatic state merging

.. code-block:: python

   # Create state type with message support
   state_type = StateTypeFactory.create_with_messages(MyState)

   # Create mergeable state type
   mergeable_type = StateTypeFactory.create_mergeable(MyState)

Discovery System
----------------

Petal provides multiple tool discovery strategies:

**DecoratorDiscovery**
   Scans modules for `@petaltool` decorated functions.

**ConfigDiscovery**
   Loads tools from YAML configuration files.

**FolderDiscovery**
   Scans directories for Python files containing tools.

**MCPDiscovery**
   Discovers tools from MCP servers with background loading.

.. code-block:: python

   from petal.core.discovery import DecoratorDiscovery, ConfigDiscovery, FolderDiscovery

   registry = ToolRegistry()
   registry.add_discovery_strategy(DecoratorDiscovery())
   registry.add_discovery_strategy(ConfigDiscovery(["config/tools.yaml"]))
   registry.add_discovery_strategy(FolderDiscovery(["tools/"]))

YAML Configuration System
-------------------------

Petal supports declarative agent configuration through YAML:

**LLMNodeConfig**
   Configuration for LLM-based nodes with provider settings.

**ReactNodeConfig**
   Configuration for ReAct reasoning nodes with tool integration.

**CustomNodeConfig**
   Configuration for custom function nodes with parameter binding.

**StateSchemaConfig**
   Dynamic state schema definition with field validation.

.. code-block:: yaml

   # llm_node.yaml
   type: llm
   name: assistant
   provider: openai
   model: gpt-4o-mini
   temperature: 0.7
   prompt: "Help with {task}"
   system_prompt: "You are a helpful assistant."

.. code-block:: python

   agent = (
       AgentFactory(DefaultState)
       .node_from_yaml("llm_node.yaml")
       .build()
   )

Builder Pattern Implementation
-----------------------------

Petal uses the Builder pattern for agent construction:

**AgentBuilder**
   Fluent interface for building agents with composition.

**AgentBuilderDirector**
   Orchestrates the complex building process using the Director pattern.

**StepConfigHandler**
   Chain of Responsibility pattern for step configuration handling.

.. code-block:: python

   # Builder pattern usage
   builder = AgentBuilder(MyState)
   director = AgentBuilderDirector(builder._config, builder._registry)
   agent = await director.build()

Plugin System
-------------

Petal provides an extensible plugin system:

**StepPlugin**
   Base class for custom step type plugins.

**Plugin Registry**
   Automatic discovery and registration of plugins.

**Custom Strategies**
   User-defined step creation strategies.

.. code-block:: python

   from petal.core.plugins.base import StepPlugin

   class MyStepPlugin(StepPlugin):
       name = "my_step"

       async def create_step(self, config: Dict[str, Any]) -> Callable:
           # Custom step creation logic
           pass

   # Plugin is automatically discovered and registered

Error Handling
--------------

Petal provides comprehensive error handling:

**Validation Errors**
   Pydantic validation for configuration objects.

**Type Errors**
   Type checking for state and function signatures.

**Discovery Errors**
   Graceful handling of missing tools and resources.

**Runtime Errors**
   Proper error propagation and context preservation.

.. code-block:: python

   try:
       agent = AgentFactory(MyState).with_chat("Hello").build()
   except ValidationError as e:
       print(f"Configuration error: {e}")
   except TypeError as e:
       print(f"Type error: {e}")

Performance Optimizations
-------------------------

Petal includes several performance optimizations:

**Lazy Loading**
   Tools and resources are loaded only when needed.

**Caching**
   Repeated operations are cached for efficiency.

**Async Operations**
   All I/O operations are async for better concurrency.

**Background Loading**
   MCP tools load in the background to avoid blocking.

.. code-block:: python

   # Background MCP loading
   factory = ToolFactory()
   factory.add_mcp("filesystem", mcp_config=config)
   # Continue with other operations while MCP loads
   await factory.await_mcp_loaded("filesystem")

Thread Safety
-------------

Petal is designed for concurrent access:

**Singleton Registries**
   Thread-safe singleton patterns for shared resources.

**Immutable Configuration**
   Configuration objects are immutable after creation.

**Async-Safe Operations**
   All operations are safe for concurrent execution.

**Lock-Free Design**
   Minimal use of locks for better performance.

.. code-block:: python

   # Thread-safe registry usage
   registry = ToolRegistry()  # Singleton
   tool = await registry.resolve("my_tool")  # Thread-safe

Integration Patterns
-------------------

Petal integrates with various external systems:

**LangGraph Integration**
   All agents can be used as LangGraph nodes.

**LangChain Compatibility**
   Compatible with LangChain tools and models.

**MCP Protocol**
   Native support for Model Context Protocol.

**Custom Integrations**
   Plugin system for custom integrations.

.. code-block:: python

   # LangGraph integration
   agent = await AgentFactory(MyState).with_chat("Hello").build()
   node = agent.as_node()  # LangGraph node

   # MCP integration
   factory = ToolFactory()
   factory.add_mcp("filesystem", mcp_config=config)

Testing Support
---------------

Petal provides comprehensive testing support:

**Mock Support**
   Easy mocking of external dependencies.

**Test Utilities**
   Helper functions for testing agents and tools.

**Isolation**
   Test isolation with registry reset capabilities.

**Coverage**
   High test coverage with comprehensive test suites.

.. code-block:: python

   # Testing with mocks
   with patch('petal.core.tool_factory.ToolFactory.resolve') as mock_resolve:
       mock_resolve.return_value = mock_tool
       agent = await AgentFactory(MyState).with_tools(["mock_tool"]).build()

Architecture Benefits
--------------------

**Modularity**
   Clear separation of concerns with well-defined interfaces.

**Extensibility**
   Plugin system allows easy addition of new features.

**Maintainability**
   Clean, well-documented code with comprehensive tests.

**Performance**
   Optimized for efficiency with async operations and caching.

**Reliability**
   Comprehensive error handling and validation.

**Developer Experience**
   Fluent APIs and excellent IDE support.

This architecture provides a solid foundation for building complex AI applications while maintaining simplicity and ease of use.
