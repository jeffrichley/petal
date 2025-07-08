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

   from petal import AgentFactory, ToolFactory

   # Create a simple tool
   @tool_fn
   def get_weather(city: str) -> str:
       """Get the current weather for a city."""
       return f"Sunny in {city}"

   # Create an agent
   agent = (AgentFactory()
       .add(get_weather)
       .with_prompt("What's the weather like in {city}?")
       .with_chat()
       .build())

   # Run the agent
   result = agent.run({"city": "San Francisco"})
   print(result)

Key Concepts
------------

AgentFactory
~~~~~~~~~~~~

The main entry point for creating agents. Provides a fluent API for configuring:

- Tools and functions
- Prompts and system messages
- LLM backends
- Memory and state management
- Logging and debugging

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
