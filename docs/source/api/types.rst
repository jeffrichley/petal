Types
=====

Type definitions and schemas for Petal framework components.

.. automodule:: petal.types
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The types module provides comprehensive type definitions for Petal framework components, including agent manifests, tool manifests, state schemas, and configuration types. These types ensure type safety and provide clear interfaces for framework components.

Key Components
--------------

- **Agent Manifests**: Type definitions for agent metadata and configuration
- **Tool Manifests**: Type definitions for tool metadata and capabilities
- **State Types**: TypedDict definitions for agent state management
- **Configuration Types**: Pydantic models for configuration validation

Agent Manifests
---------------

Agent manifests define the metadata and configuration for agents:

.. code-block:: python

    from petal.types.agent_manifest import AgentManifest

    manifest = AgentManifest(
        name="my_agent",
        description="A helpful AI assistant",
        version="1.0.0",
        author="John Doe",
        tags=["assistant", "helpful"],
        capabilities=["chat", "tools", "reasoning"],
        state_schema={
            "messages": "list",
            "user_input": "str",
            "context": "dict"
        }
    )

Tool Manifests
--------------

Tool manifests define the metadata and capabilities for tools:

.. code-block:: python

    from petal.types.tool_manifest import ToolManifest

    manifest = ToolManifest(
        name="calculator:add",
        description="Add two numbers together",
        version="1.0.0",
        author="Math Tools Inc",
        tags=["math", "arithmetic"],
        parameters={
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"}
        },
        returns={"type": "number", "description": "Sum of the numbers"}
    )

State Types
-----------

State types define the structure of agent state using TypedDict:

.. code-block:: python

    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class MyState(TypedDict):
        messages: Annotated[list, add_messages]
        user_input: str
        context: dict
        processed: bool

    # Use with AgentFactory
    agent = AgentFactory(MyState).with_chat("Hello").build()

Configuration Types
-------------------

Configuration types provide validated configuration schemas:

.. code-block:: python

    from petal.core.config.agent import AgentConfig, LLMConfig, MemoryConfig

    config = AgentConfig(
        name="my_agent",
        state_type=MyState,
        steps=[
            StepConfig(
                strategy_type="llm",
                config={"prompt_template": "Hello {name}"}
            )
        ],
        llm_config=LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7
        ),
        memory=MemoryConfig(
            memory_type="conversation",
            max_tokens=1000
        )
    )

Type Validation
---------------

All types include comprehensive validation:

.. code-block:: python

    from pydantic import ValidationError

    try:
        config = AgentConfig(
            name="",  # Invalid: empty name
            state_type=dict,
            steps=[]
        )
    except ValidationError as e:
        print(f"Configuration error: {e}")

Integration Examples
-------------------

**Agent with Custom State**
    .. code-block:: python

        from typing import Annotated, TypedDict
        from langgraph.graph.message import add_messages
        from petal.core.factory import AgentFactory

        class CustomState(TypedDict):
            messages: Annotated[list, add_messages]
            user_input: str
            personality: str
            task: str

        agent = (
            AgentFactory(CustomState)
            .with_chat(
                prompt_template="Complete task: {task}",
                system_prompt="You are a {personality} assistant."
            )
            .build()
        )

        result = await agent.arun({
            "user_input": "Hello",
            "personality": "friendly",
            "task": "Help the user",
            "messages": []
        })

**Tool with Manifest**
    .. code-block:: python

        from petal.core.decorators import petaltool
        from petal.types.tool_manifest import ToolManifest

        @petaltool("math:add")
        def add(a: float, b: float) -> float:
            """Add two numbers together."""
            return a + b

        # Tool manifest is automatically generated
        manifest = ToolManifest(
            name="math:add",
            description="Add two numbers together",
            parameters={
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            returns={"type": "number"}
        )

**Configuration with Validation**
    .. code-block:: python

        from petal.core.config.agent import AgentConfig, LLMConfig

        config = AgentConfig(
            name="validated_agent",
            state_type=MyState,
            llm_config=LLMConfig(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1000
            )
        )

        # Configuration is validated automatically
        agent = AgentFactory(config.state_type).with_chat("Hello").build()

Type Safety Features
--------------------

- **Comprehensive Type Hints**: All functions and classes include type hints
- **Pydantic Validation**: Configuration objects use Pydantic for runtime validation
- **TypedDict Support**: State types use TypedDict for compile-time checking
- **Generic Types**: Support for generic types and type parameters
- **Union Types**: Support for multiple valid types where appropriate

Error Handling
--------------

Type validation provides clear error messages:

**Invalid Configuration**
    .. code-block:: python

        try:
            config = LLMConfig(
                provider="invalid_provider",  # Not in allowed list
                model="gpt-4o-mini"
            )
        except ValidationError as e:
            print(f"Invalid provider: {e}")

**Invalid State Type**
    .. code-block:: python

        try:
            agent = AgentFactory(None)  # Invalid state type
        except TypeError as e:
            print(f"Invalid state type: {e}")

**Missing Required Fields**
    .. code-block:: python

        try:
            config = AgentConfig()  # Missing required fields
        except ValidationError as e:
            print(f"Missing fields: {e}")

Performance Considerations
-------------------------

- **Lazy Validation**: Validation occurs only when needed
- **Type Caching**: Repeated type lookups are cached
- **Efficient Serialization**: Fast serialization/deserialization
- **Memory Optimization**: Efficient memory usage for large configurations

Best Practices
--------------

1. **Use TypedDict for State**: Always define state types using TypedDict
2. **Validate Configuration**: Use Pydantic models for configuration validation
3. **Include Type Hints**: Add type hints to all functions and classes
4. **Use Annotations**: Use Annotated for special field behaviors
5. **Document Types**: Include docstrings for all type definitions
