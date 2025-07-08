Architecture Overview
=====================

This document provides an overview of the Petal framework architecture and design principles.

Design Philosophy
-----------------

Petal is built around three core principles:

1. **Pythonic API**: Fluent, chainable interfaces that feel natural to Python developers
2. **Zero-Config Discovery**: Automatic tool and agent discovery with sensible defaults
3. **LangGraph Compatibility**: All components can be used as LangGraph nodes

Core Components
---------------

AgentFactory
~~~~~~~~~~~~

The main entry point for creating agents. Encapsulates:

- **Steps**: Individual functions that process state
- **Prompts**: Template strings for LLM interactions
- **Tools**: Callable functions available to the agent
- **Configuration**: LLM settings, memory, logging, etc.

.. code-block:: python

   agent = (AgentFactory()
       .add(my_function)
       .with_prompt("Process {input}")
       .with_chat()
       .with_memory()
       .build())

ToolFactory
~~~~~~~~~~~

Manages tool discovery and registration:

- **Auto-discovery**: Scans directories for `@tool_fn` decorated functions
- **Lazy loading**: Tools are loaded on-demand
- **MCP support**: Handles Model Context Protocol tool resolution
- **Namespace support**: Tools can be namespaced (e.g., `mcp:toolname`)

.. code-block:: python

   tools = ToolFactory()
   tools.discover("tools/")
   tools.add_lazy("external:api", resolve_external_api)

GraphFactory
~~~~~~~~~~~~

Composes agents into complex workflows:

- **Node management**: Adds agents as LangGraph nodes
- **Edge definition**: Connects nodes with data flow
- **State management**: Handles state passing between nodes
- **Conditional logic**: Supports branching and loops

.. code-block:: python

   graph = (GraphFactory()
       .add_agent("research", research_agent)
       .add_agent("write", write_agent)
       .connect("research", "write")
       .build())

State Management
----------------

Petal uses a dictionary-based state system:

- **Input state**: Initial data passed to the agent
- **Step state**: Data processed by each step
- **Output state**: Final result from the agent

State flows through the system and can be modified by:

- **Steps**: Functions that receive and return state
- **Tools**: External functions that can read/write state
- **LLM**: Can access and modify state through prompts

.. code-block:: python

   @agent_step
   def process_user_input(state):
       user_input = state['input']
       processed = analyze_input(user_input)
       return {'analysis': processed}

Memory & Persistence
--------------------

Agents can be configured with different memory backends:

- **Session memory**: Per-run state persistence
- **Conversation memory**: Multi-turn dialogue history
- **Vector memory**: Semantic search over past interactions
- **Custom memory**: User-defined memory implementations

.. code-block:: python

   agent = (AgentFactory()
       .with_memory("conversation")
       .with_memory("vector", collection="user_chats")
       .build())

Tool Integration
----------------

Tools are integrated through several mechanisms:

1. **Direct registration**: Functions added via `.add()`
2. **Auto-discovery**: Scanned from directories
3. **Lazy loading**: Resolved on-demand
4. **MCP proxies**: Deferred resolution through Model Context Protocol

.. code-block:: python

   # Direct registration
   agent.add(my_tool)

   # Auto-discovery
   agent.with_tool_registry(ToolFactory().discover("tools/"))

   # MCP integration
   agent.with_mcp_proxy("mcp://localhost:3000")

Error Handling & Resilience
---------------------------

Petal provides several error handling mechanisms:

- **Retry logic**: Automatic retries with exponential backoff
- **Timeout handling**: Configurable timeouts for operations
- **Fallback strategies**: Alternative paths when steps fail
- **Error recovery**: Graceful degradation when tools are unavailable

.. code-block:: python

   agent = (AgentFactory()
       .with_retry(3, backoff_factor=2)
       .with_timeout(30)
       .with_fallback(fallback_function)
       .build())

Performance & Optimization
--------------------------

The framework is designed for performance:

- **Lazy evaluation**: Tools and steps are evaluated only when needed
- **Caching**: Results can be cached to avoid recomputation
- **Parallel execution**: Steps can run in parallel when possible
- **Resource management**: Automatic cleanup of resources

Future Extensions
-----------------

The architecture supports several planned extensions:

- **YAML configuration**: Declarative agent definitions
- **Visual debugging**: Graph visualization tools
- **Plugin system**: Third-party extensions
- **Distributed execution**: Multi-node agent workflows
