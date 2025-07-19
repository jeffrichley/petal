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

For local development setup:

.. code-block:: bash

   git clone https://github.com/jeffrichley/petal.git
   cd petal
   python scripts/setup_dev.py
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv run make test  # Verify installation
   uv run make checkit  # Run all quality checks

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

Local LLM Setup (Optional)
--------------------------

To use local LLMs via Ollama:

1. **Install Ollama**: `https://ollama.ai/ <https://ollama.ai/>`_
2. **Pull a model**: ``ollama pull llama2``
3. **Start Ollama**: ``ollama serve``
4. **Test the demo**: ``python examples/ollama_demo.py``

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
       "response": "",
       "messages": []
   })
   print(result["messages"][-1].content)

Tool Integration
----------------

Petal provides seamless tool integration with automatic discovery:

.. code-block:: python

   from petal.core.factory import AgentFactory, DefaultState
   from petal.core.decorators import petaltool

   # Define a custom tool
   @petaltool("calculator:add")
   def add_numbers(a: float, b: float) -> float:
       """Add two numbers together."""
       return a + b

   # Create agent with tools
   agent = (
       AgentFactory(DefaultState)
       .with_chat(
           prompt_template="Calculate {expression} for me.",
           system_prompt="You are a helpful math assistant."
       )
       .with_tools(["calculator:add"])
       .build()
   )

   # Run the agent
   result = await agent.arun({
       "name": "User",
       "expression": "5 + 3",
       "messages": []
   })
   print(result["messages"][-1].content)

MCP Integration
---------------

Petal supports Model Context Protocol (MCP) for external tool integration:

.. code-block:: python

   from petal.core.factory import AgentFactory, DefaultState
   from petal.core.tool_factory import ToolFactory

   # Configure MCP tools
   tool_factory = ToolFactory()
   mcp_config = {
       "filesystem": {
           "command": "mcp-server-filesystem",
           "args": ["--config", "config.json"]
       }
   }
   tool_factory.add_mcp("filesystem", mcp_config=mcp_config)

   # Wait for MCP tools to load
   await tool_factory.await_mcp_loaded("filesystem")

   # Create agent with MCP tools
   agent = (
       AgentFactory(DefaultState)
       .with_chat(
           prompt_template="Read the file at {file_path}",
           system_prompt="You are a helpful file assistant."
       )
       .with_tools(["mcp:filesystem:read_file"])
       .build()
   )

   result = await agent.arun({
       "name": "User",
       "file_path": "/etc/hosts",
       "messages": []
   })

ReAct Agent with Tools
----------------------

Create a ReAct-style agent for complex reasoning:

.. code-block:: python

   from petal.core.factory import AgentFactory
   from typing import Annotated, TypedDict
   from langgraph.graph.message import add_messages

   class ReActState(TypedDict):
       messages: Annotated[list, add_messages]
       question: str
       scratchpad: list

   agent = (
       AgentFactory(ReActState)
       .with_react_loop(
           tools=["calculator:add", "web_search"],
           reasoning_prompt="Think step by step about how to answer the question.",
           system_prompt="You are a helpful assistant that can use tools."
       )
       .build()
   )

   result = await agent.arun({
       "question": "What is 15 + 27?",
       "messages": [],
       "scratchpad": []
   })

Structured Output
-----------------

Bind Pydantic models to LLM responses for structured data:

.. code-block:: python

   from petal.core.factory import AgentFactory, DefaultState
   from pydantic import BaseModel
   from typing import List

   class Person(BaseModel):
       name: str
       age: int
       hobbies: List[str]

   agent = (
       AgentFactory(DefaultState)
       .with_chat(
           prompt_template="Extract information about {description}",
           system_prompt="You are a helpful assistant that extracts structured information."
       )
       .with_structured_output(Person)
       .build()
   )

   result = await agent.arun({
       "name": "User",
       "description": "John is 25 years old and enjoys hiking and reading",
       "messages": []
   })

   # result["person"] will be a Person instance
   print(result["person"].name)  # "John"
   print(result["person"].age)   # 25

Structured Output with Custom Key
---------------------------------

You can specify a custom key for structured output:

.. code-block:: python

   from petal.core.factory import AgentFactory, DefaultState
   from pydantic import BaseModel

   class AnalysisResult(BaseModel):
       sentiment: str
       confidence: float
       keywords: list[str]

   agent = (
       AgentFactory(DefaultState)
       .with_chat(
           prompt_template="Analyze the sentiment of: {text}",
           system_prompt="You are a sentiment analysis expert."
       )
       .with_structured_output(AnalysisResult, key="analysis")
       .build()
   )

   result = await agent.arun({
       "name": "User",
       "text": "I love this product! It's amazing!",
       "messages": []
   })

   # result["analysis"] contains the structured output
   print(result["analysis"].sentiment)  # "positive"
   print(result["analysis"].confidence)  # 0.95

YAML Configuration
------------------

Petal supports declarative agent configuration through YAML:

.. code-block:: yaml

   # agent.yaml
   type: llm
   name: assistant
   provider: openai
   model: gpt-4o-mini
   temperature: 0.7
   prompt: "Help with {task}"
   system_prompt: "You are a helpful assistant."

.. code-block:: python

   from petal.core.factory import AgentFactory, DefaultState

   agent = (
       AgentFactory(DefaultState)
       .node_from_yaml("agent.yaml")
       .build()
   )

   result = await agent.arun({
       "name": "User",
       "task": "Write a Python function",
       "messages": []
   })

Tool Discovery
--------------

Configure automatic tool discovery:

.. code-block:: python

   from petal.core.factory import AgentFactory, DefaultState

   agent = (
       AgentFactory(DefaultState)
       .with_chat(
           prompt_template="Use available tools to help with {task}",
           system_prompt="You are a helpful assistant with access to tools."
       )
       .with_tool_discovery(
           enabled=True,
           folders=["tools/"],
           config_locations=["config/tools.yaml"],
           exclude_patterns=["*_test.py"]
       )
       .build()
   )

   result = await agent.arun({
       "name": "User",
       "task": "Calculate the area of a circle with radius 5",
       "messages": []
   })

Tool Integration with Scratchpad
--------------------------------

Use ReAct-style tools with persistent scratchpad:

.. code-block:: python

   from petal.core.factory import AgentFactory, DefaultState

   agent = (
       AgentFactory(DefaultState)
       .with_chat(
           prompt_template="Help me with {task}",
           system_prompt="You are a helpful assistant that can use tools."
       )
       .with_react_tools(
           tools=["calculator:add", "web_search", "file_reader"],
           scratchpad_key="tool_observations"
       )
       .build()
   )

   result = await agent.arun({
       "name": "User",
       "task": "Calculate the population density of New York City",
       "messages": []
   })

   # Tool observations are stored in the scratchpad
   print(result["tool_observations"])

Error Handling
--------------

Petal provides comprehensive error handling:

.. code-block:: python

   from petal.core.factory import AgentFactory, DefaultState

   # Handle missing state variables
   try:
       agent = (
           AgentFactory(DefaultState)
           .with_chat(
               prompt_template="Hello {missing_variable}!",
               system_prompt="You are helpful."
           )
           .build()
       )
       result = await agent.arun({"name": "Alice", "messages": []})
   except ValueError as e:
       print(f"Template error: {e}")

   # Handle invalid LLM configuration
   try:
       agent = (
           AgentFactory(DefaultState)
           .with_chat(
               prompt_template="Hello {name}!",
               llm_config={
                   "provider": "invalid_provider",
                   "model": "gpt-4o-mini"
               }
           )
           .build()
       )
   except ValueError as e:
       print(f"LLM config error: {e}")

   # Handle missing tools
   try:
       agent = (
           AgentFactory(DefaultState)
           .with_chat("Hello")
           .with_tools(["nonexistent_tool"])
           .build()
       )
   except KeyError as e:
       print(f"Tool not found: {e}")

Performance Optimization
------------------------

Optimize your agents for better performance:

.. code-block:: python

   from petal.core.factory import AgentFactory, DefaultState

   # Use caching for repeated operations
   agent = (
       AgentFactory(DefaultState)
       .with_chat(
           prompt_template="Process {input}",
           system_prompt="You are efficient."
       )
       .with_llm(
           provider="openai",
           model="gpt-4o-mini",
           temperature=0.0,  # Lower temperature for consistency
           max_tokens=100    # Limit tokens for speed
       )
       .build()
   )

   # Batch process multiple inputs
   inputs = [
       {"name": "Alice", "input": "Hello", "messages": []},
       {"name": "Bob", "input": "Hi there", "messages": []},
       {"name": "Charlie", "input": "Good morning", "messages": []}
   ]

   results = []
   for input_data in inputs:
       result = await agent.arun(input_data)
       results.append(result)

Integration with External Systems
--------------------------------

Integrate with external APIs and systems:

.. code-block:: python

   from petal.core.factory import AgentFactory, DefaultState
   import asyncio

   async def fetch_external_data(state: dict) -> dict:
       # Simulate external API call
       await asyncio.sleep(0.1)
       state["external_data"] = {"status": "success", "data": "external info"}
       return state

   async def process_with_external_data(state: dict) -> dict:
       if "external_data" in state:
           state["processed"] = True
       return state

   agent = (
       AgentFactory(DefaultState)
       .add(fetch_external_data)
       .add(process_with_external_data)
       .with_chat(
           prompt_template="Process data: {external_data}",
           system_prompt="You are a data processor."
       )
       .build()
   )

   result = await agent.arun({
       "name": "User",
       "messages": []
   })

Next Steps
----------

Now that you have the basics, explore:

- **API Reference**: Detailed documentation of all classes and methods
- **Examples**: Comprehensive examples and tutorials
- **Architecture**: Understanding the framework design
- **Advanced Features**: MCP integration, custom steps, and more

For more complex examples and advanced usage patterns, see the :doc:`examples/index` section.
