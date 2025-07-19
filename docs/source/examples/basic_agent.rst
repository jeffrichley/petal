Basic Agent
==========

This example demonstrates how to create a simple agent using both AgentFactory and AgentBuilder.

AgentFactory Example
-------------------

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState

    # Create a simple agent with LLM support
    agent = (
        AgentFactory(DefaultState)
        .with_chat(
            prompt_template="Hello {name}! How can I help you today?",
            system_prompt="You are a helpful and friendly assistant."
        )
        .build()
    )

    # Run the agent
    result = await agent.arun({"name": "Alice", "messages": []})
    print(result["messages"][-1].content)

AgentBuilder Example
-------------------

.. code-block:: python

    from petal.core.builders.agent import AgentBuilder
    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class MyState(TypedDict):
        messages: Annotated[list, add_messages]
        user_input: str

    # Build agent with explicit configuration
    builder = AgentBuilder(MyState)
    agent = (
        await (
            builder.with_step(
                "llm",
                prompt_template="User says: {user_input}. Respond helpfully."
            )
            .with_system_prompt("You are a helpful assistant.")
            .with_llm(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=150
            )
            .build()
        )
    )

    # Run the agent
    result = await agent.arun({
        "user_input": "Hello! How are you today?",
        "messages": []
    })
    print(result["messages"][-1].content)

Custom Step Example
------------------

.. code-block:: python

    from petal.core.builders.agent import AgentBuilder
    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class CustomState(TypedDict):
        messages: Annotated[list, add_messages]
        user_input: str
        processed: bool

    async def custom_step(state: dict) -> dict:
        """Custom step that processes the input."""
        return {
            "processed": True,
            "input_length": len(state["user_input"])
        }

    # Build agent with custom step
    builder = AgentBuilder(CustomState)
    agent = (
        await (
            builder.with_step("custom", step_function=custom_step)
            .with_step(
                "llm",
                prompt_template="Processed input length: {input_length}. User says: {user_input}"
            )
            .with_system_prompt("You are a helpful assistant.")
            .with_llm(provider="openai", model="gpt-4o-mini")
            .build()
        )
    )

    # Run the agent
    result = await agent.arun({
        "user_input": "Hello world!",
        "processed": False,
        "messages": []
    })
    print(result["messages"][-1].content)

Local LLM Example
----------------

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState

    # Create agent with local LLM via Ollama
    agent = (
        AgentFactory(DefaultState)
        .with_chat(
            prompt_template="Hello {name}! How can I help you today?",
            system_prompt="You are a helpful and friendly assistant.",
            llm_config={
                "provider": "ollama",
                "model": "llama2",
                "temperature": 0.7,
            }
        )
        .build()
    )

    # Run the agent
    result = await agent.arun({"name": "Alice", "messages": []})
    print(result["messages"][-1].content)

Key Differences
--------------

- **AgentFactory**: Higher-level API with automatic step management and fluent interface
- **AgentBuilder**: Lower-level API with explicit step configuration and more control
- Both support the same core functionality but with different levels of abstraction
- All operations are async-first for better performance
- Both APIs support local LLMs via Ollama and cloud providers

State Management
----------------

Both APIs use TypedDict for strongly-typed state:

.. code-block:: python

    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class MyState(TypedDict):
        messages: Annotated[list, add_messages]  # Auto-merging messages
        user_input: str
        context: dict
        metadata: dict

    # State flows through the system
    result = await agent.arun({
        "user_input": "Hello",
        "context": {},
        "metadata": {},
        "messages": []
    })
