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

AgentFactory (High-level API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main entry point for creating agents. Provides a fluent interface that uses AgentBuilder internally:

- **Steps**: Individual functions that process state
- **Prompts**: Template strings for LLM interactions
- **System Prompts**: Dynamic system prompts with state variable interpolation
- **Configuration**: LLM settings, memory, logging, etc.

.. code-block:: python

   agent = (AgentFactory(DefaultState)
       .add(my_function)
       .with_chat(
           prompt_template="Process {input}",
           system_prompt="You are a helpful assistant."
       )
       .build())

AgentBuilder (Lower-level API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provides explicit control over agent building using composition patterns:

- **Step Configuration**: Explicit step type and configuration
- **Named Parameters**: Type-safe LLM configuration with named parameters
- **Configuration Objects**: Direct access to Pydantic configuration models
- **Step Registry**: Pluggable step strategies

.. code-block:: python

   builder = AgentBuilder(MyState)
   agent = (
       builder.with_step("custom", step_function=my_function)
       .with_step("llm", prompt_template="Process {input}")
       .with_system_prompt("You are a helpful assistant.")
       .with_llm(provider="openai", model="gpt-4o-mini")
       .build()
   )

Step Strategy Pattern
--------------------

The framework uses the Strategy pattern for step management:

**StepStrategy ABC**
   Abstract base class defining the interface for step creation strategies.

**LLMStepStrategy**
   Strategy for creating LLM steps with prompt templates and system prompts.

**CustomStepStrategy**
   Strategy for creating custom function steps.

**StepRegistry**
   Registry for managing and discovering step strategies.

.. code-block:: python

   # Register a custom step strategy
   registry = StepRegistry()
   registry.register("my_step", MyStepStrategy)

   # Use the strategy
   strategy = registry.get_strategy("my_step")
   step = strategy.create_step(config)

Configuration Management
-----------------------

Petal uses Pydantic models for configuration management:

**AgentConfig**
   Main configuration object containing all agent settings.

**StepConfig**
   Configuration for individual steps.

**LLMConfig**
   LLM-specific configuration with named parameters.

**MemoryConfig**
   Memory backend configuration.

**GraphConfig**
   LangGraph configuration options.

**LoggingConfig**
   Logging configuration.

.. code-block:: python

   config = AgentConfig(
       state_type=MyState,
       steps=[StepConfig(strategy_type="llm", config={...})],
       llm_config=LLMConfig(provider="openai", model="gpt-4o-mini"),
       memory=MemoryConfig(memory_type="conversation"),
       graph_config=GraphConfig(topology="linear"),
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

.. code-block:: python

   # Create state type with message support
   state_type = StateTypeFactory.create_with_messages(MyState)

   # Create mergeable state type
   mergeable_type = StateTypeFactory.create_mergeable(MyState)

AgentBuilderDirector
--------------------

Implements the Director pattern for agent building:

- **Build orchestration**: Coordinates the building process
- **State type creation**: Uses StateTypeFactory for state types
- **Graph building**: Creates LangGraph StateGraph
- **Validation**: Validates configuration before building

.. code-block:: python

   director = AgentBuilderDirector(config, registry)
   agent = director.build()

Memory & Persistence
--------------------

Agents can be configured with different memory backends:

- **Session memory**: Per-run state persistence
- **Conversation memory**: Multi-turn dialogue history
- **Vector memory**: Semantic search over past interactions
- **Custom memory**: User-defined memory implementations

.. code-block:: python

   builder = AgentBuilder(MyState)
   agent = (
       builder.with_memory({
           "memory_type": "conversation",
           "max_tokens": 1000
       })
       .build()
   )

Tool Integration
----------------

Tools are integrated through several mechanisms:

1. **Direct registration**: Functions added via `.add()` (AgentFactory) or `.with_step()` (AgentBuilder)
2. **Custom steps**: Arbitrary functions that process state
3. **LLM integration**: Tools can be called from LLM prompts
4. **MCP proxies**: Deferred resolution through Model Context Protocol

.. code-block:: python

   # Direct registration with AgentFactory
   agent.add(my_tool)

   # Custom step with AgentBuilder
   builder.with_step("custom", step_function=my_tool)

Error Handling & Resilience
---------------------------

Petal provides several error handling mechanisms:

- **Configuration validation**: Pydantic models validate configuration
- **Step validation**: Step strategies validate their configurations
- **State validation**: TypedDict ensures state type safety
- **Error recovery**: Graceful degradation when steps fail

.. code-block:: python

   # Configuration validation
   try:
       llm_config = LLMConfig(provider="openai", model="gpt-4o-mini")
   except ValidationError as e:
       print(f"Invalid configuration: {e}")

   # Step validation
   try:
       strategy = registry.get_strategy("unknown_step")
   except ValueError as e:
       print(f"Unknown step type: {e}")

Performance & Optimization
--------------------------

The framework is designed for performance:

- **Type caching**: StateTypeFactory caches created types
- **Lazy evaluation**: Steps are created only when needed
- **Configuration validation**: Early validation prevents runtime errors
- **Memory management**: Automatic cleanup of resources

Future Extensions
-----------------

The architecture supports several planned extensions:

- **Plugin system**: Third-party step strategies
- **Visual debugging**: Graph visualization tools
- **YAML configuration**: Declarative agent definitions
- **Distributed execution**: Multi-node agent workflows
