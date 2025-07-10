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

    # Run the agent
    result = await agent.arun({
        "user_input": "Hello! How are you today?",
        "messages": []
    })
    print(result["messages"][-1].content)

Key Differences
--------------

- **AgentFactory**: Higher-level API with automatic step management
- **AgentBuilder**: Lower-level API with explicit step configuration
- Both support the same core functionality but with different levels of control
