API Reference
=============

This section contains the complete API reference for the Petal framework.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   agent_factory
   agent_builder
   decorators
   registry
   tool_factory
   graph_factory
   types

Core Classes
------------

.. automodule:: petal.core.factory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.builders.agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.builders.director
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.tool_factory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.registry
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.decorators
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Classes
--------------------

.. automodule:: petal.core.config.agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.config.state
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.config.yaml
   :members:
   :undoc-members:
   :show-inheritance:

Step Strategies
--------------

.. automodule:: petal.core.steps.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.steps.llm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.steps.custom
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.steps.registry
   :members:
   :undoc-members:
   :show-inheritance:

YAML Configuration
-----------------

.. automodule:: petal.core.yaml.parser
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.yaml.handlers.factory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.yaml.handlers.llm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.yaml.handlers.react
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.yaml.handlers.custom
   :members:
   :undoc-members:
   :show-inheritance:

Plugin System
------------

.. automodule:: petal.core.plugins.base
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Handlers
---------------------

.. automodule:: petal.core.builders.handlers.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.builders.handlers.llm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.core.builders.handlers.custom
   :members:
   :undoc-members:
   :show-inheritance:

Agent and State Types
--------------------

.. automodule:: petal.core.agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.types.agent_manifest
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.types.tool_manifest
   :members:
   :undoc-members:
   :show-inheritance:

Utility Modules
--------------

.. automodule:: petal.core.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: petal.config.settings
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Petal API is organized into several key areas:

**Core Factory Classes**
   High-level interfaces for creating agents and tools.

**Builder Classes**
   Lower-level interfaces for fine-grained control over agent construction.

**Configuration Classes**
   Pydantic models for type-safe configuration management.

**Step Strategies**
   Pluggable strategies for different types of agent steps.

**YAML Configuration**
   Support for declarative agent configuration.

**Plugin System**
   Extensible system for custom functionality.

**Utility Modules**
   Helper functions and utilities.

Key Design Patterns
-------------------

**Builder Pattern**
   Used throughout the framework for fluent, chainable interfaces.

**Strategy Pattern**
   Applied to step creation and tool discovery.

**Singleton Pattern**
   Used for registries and shared resources.

**Factory Pattern**
   Central to agent and tool creation.

**Chain of Responsibility**
   Used for configuration handling.

Usage Examples
--------------

**Basic Agent Creation**
   .. code-block:: python

      from petal.core.factory import AgentFactory, DefaultState

      agent = (
          AgentFactory(DefaultState)
          .with_chat("Hello {name}!")
          .build()
      )

**Advanced Agent with Tools**
   .. code-block:: python

      from petal.core.factory import AgentFactory, DefaultState
      from petal.core.tool_factory import ToolFactory

      # Configure MCP tools
      tool_factory = ToolFactory()
      tool_factory.add_mcp("filesystem", mcp_config=config)
      await tool_factory.await_mcp_loaded("filesystem")

      agent = (
          AgentFactory(DefaultState)
          .with_chat("Help with {task}")
          .with_tools(["mcp:filesystem:read_file"])
          .build()
      )

**Custom Step with AgentBuilder**
   .. code-block:: python

      from petal.core.builders.agent import AgentBuilder

      async def custom_step(state: dict) -> dict:
          return {"processed": True}

      builder = AgentBuilder(MyState)
      agent = (
          await (
              builder.with_step("custom", step_function=custom_step)
              .with_step("llm", prompt_template="Process {input}")
              .build()
          )
      )

**YAML Configuration**
   .. code-block:: python

      from petal.core.factory import AgentFactory, DefaultState

      agent = (
          AgentFactory(DefaultState)
          .node_from_yaml("agent.yaml")
          .build()
      )

Error Handling
--------------

All API methods include comprehensive error handling:

**Validation Errors**
   Pydantic validation for configuration objects.

**Type Errors**
   Type checking for function signatures and state.

**Discovery Errors**
   Graceful handling of missing tools and resources.

**Runtime Errors**
   Proper error propagation with context.

Thread Safety
-------------

The API is designed for concurrent access:

**Singleton Registries**
   Thread-safe singleton patterns.

**Immutable Configuration**
   Configuration objects are immutable after creation.

**Async-Safe Operations**
   All operations are safe for concurrent execution.

Performance Considerations
-------------------------

**Lazy Loading**
   Resources are loaded only when needed.

**Caching**
   Repeated operations are cached for efficiency.

**Async Operations**
   All I/O operations are async for better concurrency.

**Background Loading**
   MCP tools load in the background to avoid blocking.
