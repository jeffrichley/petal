Petal Documentation
===================

Welcome to the Petal documentation! Petal is an elegant, opinionated agent and tool creation framework for building modular LLM systems using LangChain, LangGraph, and Pydantic.

**Petal** provides a fluent, chainable API for creating AI agents with automatic tool discovery, strong typing, and seamless LangGraph integration. It's designed to help you build powerful, discoverable agents and tools with minimal boilerplate.

Key Features
-----------

- **🔗 Fluent Chaining API**: Compose agents with readable, chainable setup flows (`.with_chat()`, `.with_prompt()`, etc.)
- **🧠 Tool & MCP Discovery**: Automatically register local and external tools using a configurable registry with multiple discovery strategies
- **🏗️ LangGraph Integration**: Agents are directly runnable as LangGraph nodes with native support for state merging and memory
- **⚙️ Factory-Based Architecture**: Easily scaffold agents and tools using expressive factory methods with IDE support
- **📄 Rich Manifest Typing**: All agents and tools are strongly typed using TypedDict for autocompletion and clarity
- **🔁 Declarative State Merging**: Fields can auto-merge (e.g., `append`, `extend`) instead of overwriting
- **🎯 Named Parameter LLM Configuration**: Use `with_llm(provider, model, temperature=0.0)` instead of magic dictionary keys
- **🏠 Local LLM Support**: Run local models via Ollama with the same interface as cloud providers
- **💬 System Prompt Support**: Add system prompts with state variable interpolation for dynamic behavior
- **🔄 ReAct Loops**: Built-in support for reasoning and tool use workflows
- **📋 YAML Configuration**: Declarative agent definitions with comprehensive validation
- **🔌 Plugin System**: Extensible step type registration and management
- **🔍 Multiple Discovery Strategies**: Decorator, config, folder, and MCP-based tool discovery
- **🛡️ Thread Safety**: Singleton patterns and thread-safe operations throughout
- **⚡ Async-First**: All operations are async-friendly for better performance
- **🎨 Structured Output**: Pydantic model binding for LLM responses
- **🔧 MCP Integration**: Native support for Model Context Protocol

Quick Start
-----------

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

Local LLM with Ollama
--------------------

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState

    # Use local LLM via Ollama
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
    print(result["messages"][-1].content)

Advanced Agent with Custom Steps
-------------------------------

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

Using AgentBuilder (Lower-level API)
-----------------------------------

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
    print(result["messages"][-1].content)

MCP Integration
---------------

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

Structured Output
-----------------

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/index
   examples/index
   architecture
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
