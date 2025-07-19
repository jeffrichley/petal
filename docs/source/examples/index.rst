Examples
========

This section contains comprehensive examples and tutorials for using Petal.

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   basic_agent
   custom_tools
   tool_integration
   graph_workflows
   memory_management
   yaml_configuration

Quick Examples
--------------

Simple Chat Agent
~~~~~~~~~~~~~~~~

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState

    agent = (
        AgentFactory(DefaultState)
        .with_chat(
            prompt_template="Hello {name}! How can I help you today?",
            system_prompt="You are a helpful and friendly assistant."
        )
        .build()
    )

    result = await agent.arun({"name": "Alice", "messages": []})
    print(result["messages"][-1].content)

Agent with Tools
~~~~~~~~~~~~~~~

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState
    from petal.core.decorators import petaltool

    @petaltool("calculator:add")
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    agent = (
        AgentFactory(DefaultState)
        .with_chat(
            prompt_template="Calculate {expression} for me.",
            system_prompt="You are a helpful math assistant."
        )
        .with_tools(["calculator:add"])
        .build()
    )

    result = await agent.arun({
        "name": "User",
        "expression": "5 + 3",
        "messages": []
    })

ReAct Agent with Reasoning Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~

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

Local LLM with Ollama
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState

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

    result = await agent.arun({"name": "Alice", "messages": []})

YAML Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # agent.yaml
    type: llm
    name: assistant
    provider: openai
    model: gpt-4o-mini
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

Custom Steps
~~~~~~~~~~~

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

MCP Integration with ToolFactory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Tool Discovery
~~~~~~~~~~~~~

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

AgentBuilder (Lower-level API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from petal.core.builders.agent import AgentBuilder
    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class MyState(TypedDict):
        messages: Annotated[list, add_messages]
        user_input: str

    async def custom_step(state: dict) -> dict:
        return {"processed": True, "input_length": len(state["user_input"])}

    builder = AgentBuilder(MyState)
    agent = (
        await (
            builder.with_step("custom", step_function=custom_step)
            .with_step(
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

    result = await agent.arun({
        "user_input": "Hello! How are you today?",
        "messages": []
    })

Structured Output with Custom Key
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Tool Integration with Scratchpad
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Advanced State Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from petal.core.factory import AgentFactory
    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class AdvancedState(TypedDict):
        messages: Annotated[list, add_messages]
        user_input: str
        context: dict
        metadata: dict
        processing_steps: list[str]

    async def add_context(state: dict) -> dict:
        state["context"]["timestamp"] = "2024-01-01T12:00:00Z"
        state["processing_steps"].append("context_added")
        return state

    async def add_metadata(state: dict) -> dict:
        state["metadata"]["user_agent"] = "Petal Framework"
        state["processing_steps"].append("metadata_added")
        return state

    agent = (
        AgentFactory(AdvancedState)
        .add(add_context)
        .add(add_metadata)
        .with_chat(
            prompt_template="Process: {user_input}",
            system_prompt="You are a helpful assistant. Context: {context}"
        )
        .build()
    )

    result = await agent.arun({
        "user_input": "Hello world",
        "context": {},
        "metadata": {},
        "processing_steps": [],
        "messages": []
    })

    print(result["processing_steps"])  # ["context_added", "metadata_added"]

Error Handling Examples
~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
