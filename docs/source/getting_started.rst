Getting Started
===============

Installation
------------

Install Petal using pip:

.. code-block:: bash

   pip install petal

Or install with development dependencies:

.. code-block:: bash

   pip install petal[dev]

Quick Start
-----------

Here's a simple example of creating an agent with Petal:

.. code-block:: python

   from petal.core.factory import AgentFactory, DefaultState

   # Create an agent with LLM support
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

Advanced Example with Custom Steps
---------------------------------

.. code-block:: python

   from petal.core.factory import AgentFactory
   from typing import Annotated, TypedDict
   from langgraph.graph.message import add_messages

   # Define custom state
   class CustomState(TypedDict):
       messages: Annotated[list, add_messages]
       name: str
       personality: str
       processed: bool

   # Custom step function
   async def set_personality(state: dict) -> dict:
       state["personality"] = "pirate"
       return state

   # Create agent with multiple steps
   agent = (
       AgentFactory(CustomState)
       .add(set_personality)
       .with_chat(
           prompt_template="The user's name is {name}. Say something nice to them.",
           system_prompt="You are a {personality} assistant."
       )
       .build()
   )

   # Run the agent
   result = await agent.arun({
       "name": "Alice",
       "personality": "",
       "processed": False,
       "messages": []
   })
   print(result["messages"][-1].content)

Using AgentBuilder (Lower-level API)
-----------------------------------

For more control over the building process, you can use AgentBuilder directly:

.. code-block:: python

   from petal.core.builders.agent import AgentBuilder
   from typing import Annotated, TypedDict
   from langgraph.graph.message import add_messages

   class MyState(TypedDict):
       messages: Annotated[list, add_messages]
       user_input: str
       response: str

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
       "response": "",
       "messages": []
   })
   print(result["messages"][-1].content)

Key Concepts
------------

AgentFactory
~~~~~~~~~~~~

The main entry point for creating agents. Provides a fluent API for configuring:

- Custom step functions with `.add()`
- LLM steps with `.with_chat()`
- Prompt templates with `.with_prompt()`
- System prompts with `.with_system_prompt()`
- State management and memory
- Logging and debugging

AgentBuilder
~~~~~~~~~~~~

Lower-level builder pattern for more control:

- Explicit step configuration with `.with_step()`
- Named parameter LLM configuration with `.with_llm()`
- Memory and logging configuration
- Graph configuration options
- Direct access to configuration objects

State Management
~~~~~~~~~~~~~~~

Petal uses TypedDict for strongly-typed state:

- Messages are automatically managed with `add_messages`
- State variables can be referenced in prompts
- System prompts support state variable interpolation
- Custom state types provide type safety

LLM Integration
~~~~~~~~~~~~~~~

Seamless integration with language models:

- Support for multiple LLM providers (OpenAI, Anthropic, etc.)
- Named parameter configuration for better IDE support
- System prompts with state variable formatting
- Automatic message handling and state management

ToolFactory
~~~~~~~~~~~

Manages tool discovery and registration:

- Auto-discovers tools from directories
- Supports lazy loading
- Handles MCP-style tool resolution

GraphFactory
~~~~~~~~~~~~

Composes agents into complex workflows:

- Wire multiple agents together
- Define execution graphs
- Support conditional logic and loops

Next Steps
----------

- :doc:`api/index` - Complete API reference
- :doc:`examples/index` - Tutorials and examples
- :doc:`architecture` - Framework architecture overview
