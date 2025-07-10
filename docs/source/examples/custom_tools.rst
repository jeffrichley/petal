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
        builder.with_step("custom", step_function=fetch_weather)
        .with_step(
            "llm",
            prompt_template="Weather data: {weather_data}. Explain it to the user."
        )
        .with_system_prompt("You are a weather assistant.")
        .with_llm(provider="openai", model="gpt-4o-mini")
        .build()
    )

    # Run the agent
    result = await agent.arun({
        "query": "What's the weather in San Francisco?",
        "weather_data": "",
        "messages": []
    })
    print(result["messages"][-1].content)

Key Points
----------

- Custom steps can be async functions that modify state
- Steps are executed in the order they're added
- State variables can be referenced in LLM prompts
- Custom tools can be integrated into the agent workflow
- Both AgentFactory and AgentBuilder support custom steps
