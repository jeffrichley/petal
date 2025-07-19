Custom Tools
===========

This example demonstrates how to create custom steps and tools using both AgentFactory and AgentBuilder.

Custom Steps with AgentFactory
-----------------------------

.. code-block:: python

    from petal.core.factory import AgentFactory
    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class CustomState(TypedDict):
        messages: Annotated[list, add_messages]
        name: str
        personality: str
        processed: bool

    # Custom step function
    async def set_personality(state: dict) -> dict:
        state["personality"] = "pirate"
        return state

    # Create agent with custom step and LLM
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

Custom Steps with AgentBuilder
-----------------------------

.. code-block:: python

    from petal.core.builders.agent import AgentBuilder
    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class MyState(TypedDict):
        messages: Annotated[list, add_messages]
        user_input: str
        processed: bool
        result: str

    # Custom step functions
    async def process_input(state: dict) -> dict:
        state["processed"] = True
        state["result"] = f"Processed: {state['user_input']}"
        return state

    async def format_response(state: dict) -> dict:
        state["result"] = f"Formatted: {state['result']}"
        return state

    # Build agent with multiple custom steps
    builder = AgentBuilder(MyState)
    agent = (
        await (
            builder.with_step("custom", step_function=process_input)
            .with_step("custom", step_function=format_response)
            .with_step(
                "llm",
                prompt_template="The processed result is: {result}. Explain it."
            )
            .with_system_prompt("You are a helpful assistant.")
            .with_llm(provider="openai", model="gpt-4o-mini")
            .build()
        )
    )

    # Run the agent
    result = await agent.arun({
        "user_input": "Hello world",
        "processed": False,
        "result": "",
        "messages": []
    })
    print(result["messages"][-1].content)

Custom Tool Functions
--------------------

You can also create custom tool functions that can be used within LLM steps:

.. code-block:: python

    from petal.core.builders.agent import AgentBuilder
    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class ToolState(TypedDict):
        messages: Annotated[list, add_messages]
        query: str
        weather_data: str

    # Custom tool function
    async def get_weather(city: str) -> str:
        """Get weather information for a city."""
        # Simulate weather API call
        return f"Sunny and 72°F in {city}"

    # Custom step that uses the tool
    async def fetch_weather(state: dict) -> dict:
        city = state["query"].split()[-1]  # Extract city from query
        state["weather_data"] = await get_weather(city)
        return state

    # Build agent with tool usage
    builder = AgentBuilder(ToolState)
    agent = (
        await (
            builder.with_step("custom", step_function=fetch_weather)
            .with_step(
                "llm",
                prompt_template="Weather data: {weather_data}. Explain it to the user."
            )
            .with_system_prompt("You are a weather assistant.")
            .with_llm(provider="openai", model="gpt-4o-mini")
            .build()
        )
    )

    # Run the agent
    result = await agent.arun({
        "query": "What's the weather in San Francisco?",
        "weather_data": "",
        "messages": []
    })
    print(result["messages"][-1].content)

Registered Tools with Decorators
-------------------------------

Create tools using the `@petaltool` decorator:

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState
    from petal.core.decorators import petaltool

    # Define a custom tool with decorator
    @petaltool("weather:get_weather")
    async def get_weather_tool(city: str) -> str:
        """Get weather information for a city."""
        # Simulate weather API call
        return f"Sunny and 72°F in {city}"

    # Create agent with registered tool
    agent = (
        AgentFactory(DefaultState)
        .with_chat(
            prompt_template="Get weather for {city}",
            system_prompt="You are a weather assistant."
        )
        .with_tools(["weather:get_weather"])
        .build()
    )

    # Run the agent
    result = await agent.arun({
        "name": "User",
        "city": "San Francisco",
        "messages": []
    })
    print(result["messages"][-1].content)

MCP Tools Integration
--------------------

Integrate MCP tools with custom processing:

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

    # Custom step for file processing
    async def process_file_info(state: dict) -> dict:
        # Add custom processing logic here
        state["file_processed"] = True
        return state

    # Create agent with MCP tools and custom processing
    agent = (
        AgentFactory(DefaultState)
        .add(process_file_info)
        .with_chat(
            prompt_template="Read and process the file at {file_path}",
            system_prompt="You are a file processing assistant."
        )
        .with_tools(["mcp:filesystem:read_file"])
        .build()
    )

    # Run the agent
    result = await agent.arun({
        "name": "User",
        "file_path": "/etc/hosts",
        "file_processed": False,
        "messages": []
    })
    print(result["messages"][-1].content)

Structured Output with Custom Tools
----------------------------------

Combine custom tools with structured output:

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState
    from petal.core.decorators import petaltool
    from pydantic import BaseModel

    class AnalysisResult(BaseModel):
        sentiment: str
        confidence: float
        keywords: list[str]

    # Custom analysis tool
    @petaltool("analysis:analyze_text")
    async def analyze_text_tool(text: str) -> dict:
        """Analyze text sentiment and extract keywords."""
        # Simulate analysis
        return {
            "sentiment": "positive",
            "confidence": 0.85,
            "keywords": ["text", "analysis", "sentiment"]
        }

    # Custom step for preprocessing
    async def preprocess_text(state: dict) -> dict:
        state["text"] = state["text"].lower().strip()
        return state

    # Create agent with custom tool and structured output
    agent = (
        AgentFactory(DefaultState)
        .add(preprocess_text)
        .with_chat(
            prompt_template="Analyze the text: {text}",
            system_prompt="You are a text analysis expert."
        )
        .with_tools(["analysis:analyze_text"])
        .with_structured_output(AnalysisResult, key="analysis")
        .build()
    )

    # Run the agent
    result = await agent.arun({
        "name": "User",
        "text": "I love this amazing product!",
        "messages": []
    })

    # Access structured output
    print(result["analysis"].sentiment)  # "positive"
    print(result["analysis"].confidence)  # 0.85

Key Points
----------

- Custom steps can be async functions that modify state
- Steps are executed in the order they're added
- State variables can be referenced in LLM prompts
- Custom tools can be integrated into the agent workflow
- Both AgentFactory and AgentBuilder support custom steps
- Tools can be registered with decorators for automatic discovery
- MCP tools can be combined with custom processing steps
- Structured output works seamlessly with custom tools
- All operations are async-first for better performance
