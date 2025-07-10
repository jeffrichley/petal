Examples
========

This section contains examples and tutorials for using the Petal framework.

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   basic_agent
   custom_tools
   graph_workflows
   memory_management
   tool_integration

Quick Examples
-------------

Basic Agent Creation
~~~~~~~~~~~~~~~~~~~

Using AgentFactory (High-level API):

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState

    agent = (
        AgentFactory(DefaultState)
        .with_chat(
            prompt_template="Hello {name}! How can I help you today?",
            system_prompt="You are a helpful assistant."
        )
        .build()
    )

    result = await agent.arun({"name": "Alice", "messages": []})
    print(result["messages"][-1].content)

Using AgentBuilder (Lower-level API):

.. code-block:: python

    from petal.core.builders.agent import AgentBuilder
    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class MyState(TypedDict):
        messages: Annotated[list, add_messages]
        user_input: str

    builder = AgentBuilder(MyState)
    agent = (
        builder.with_step(
            "llm",
            prompt_template="User says: {user_input}. Respond helpfully."
        )
        .with_system_prompt("You are a helpful assistant.")
        .with_llm(provider="openai", model="gpt-4o-mini")
        .build()
    )

    result = await agent.arun({
        "user_input": "Hello! How are you today?",
        "messages": []
    })
    print(result["messages"][-1].content)

Custom Steps
~~~~~~~~~~~

Adding custom processing steps:

.. code-block:: python

    from petal.core.factory import AgentFactory
    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class CustomState(TypedDict):
        messages: Annotated[list, add_messages]
        name: str
        personality: str

    async def set_personality(state: dict) -> dict:
        state["personality"] = "pirate"
        return state

    agent = (
        AgentFactory(CustomState)
        .add(set_personality)
        .with_chat(
            prompt_template="The user's name is {name}. Say something nice to them.",
            system_prompt="You are a {personality} assistant."
        )
        .build()
    )

    result = await agent.arun({
        "name": "Alice",
        "personality": "",
        "messages": []
    })
    print(result["messages"][-1].content)

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~

Using named parameters for LLM configuration:

.. code-block:: python

    from petal.core.builders.agent import AgentBuilder
    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class DemoState(TypedDict):
        messages: Annotated[list, add_messages]
        user_input: str
        response: str

    builder = AgentBuilder(DemoState)
    agent = (
        builder.with_step(
            "llm",
            prompt_template="User says: {user_input}. Respond helpfully."
        )
        .with_llm(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,  # More creative responses
            max_tokens=150,   # Shorter responses
        )
        .build()
    )

    result = await agent.arun({
        "user_input": "Hello! How are you today?",
        "response": "",
        "messages": []
    })
    print(result["messages"][-1].content)

Available Examples
-----------------

:doc:`basic_agent`
   Simple agent creation with both AgentFactory and AgentBuilder APIs.

:doc:`custom_tools`
   Creating custom steps and tool functions for agent workflows.

:doc:`graph_workflows`
   Building complex multi-agent workflows with LangGraph.

:doc:`memory_management`
   Managing conversation history and agent memory.

:doc:`tool_integration`
   Integrating external tools and APIs with agents.

Running Examples
---------------

All examples can be run from the project root:

.. code-block:: bash

    # Run basic playground
    python examples/playground.py

    # Run advanced playground with Rich logging
    python examples/playground2.py

    # Run improved API demo
    python examples/improved_api_demo.py

    # Run custom tool example
    python examples/custom_tool.py
