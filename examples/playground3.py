"""
Playground demonstrating tool integration with AgentFactory.

This example shows:
1. Basic tool usage with AgentFactory
2. ReAct-style tools with scratchpad
3. Tool registration and resolution
4. End-to-end tool execution flow
"""

import asyncio
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from petal import AgentFactory
from typing_extensions import TypedDict

load_dotenv()


# Define our state schema
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# Create some example tools
@tool
async def calculator(query: str) -> str:
    """A simple calculator that can perform basic math operations."""
    try:
        # Simple eval for demo - in production, use a safer math parser
        result = eval(query)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
async def weather_tool(city: str) -> str:
    """Get weather information for a city (simulated)."""
    # Simulated weather data
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Rainy, 55°F",
        "Tokyo": "Cloudy, 68°F",
        "Sydney": "Clear, 75°F",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


@tool
async def echo_tool(message: str) -> str:
    """Echo back the input message."""
    return f"Echo: {message}"


async def main():
    """Demonstrate tool integration with AgentFactory."""

    print("🤖 Petal AgentFactory Tool Integration Demo")
    print("=" * 50)

    # Create factory and register tools using ToolRegistry singleton
    factory = AgentFactory(ChatState)
    from petal.core.registry import ToolRegistry

    registry = ToolRegistry()
    registry.add("calculator", calculator)
    registry.add("weather_tool", weather_tool)
    registry.add("echo_tool", echo_tool)

    print("\n1️⃣ Basic Tool Usage")
    print("-" * 30)

    # Build agent with tools using LLM types
    agent = await (
        factory.with_chat(llm_config={"provider": "openai", "model": "gpt-4o-mini"})
        .with_tools(["calculator", "weather_tool"])
        .build()
    )

    # Render the agent's graph
    AgentFactory.diagram_agent(agent, "agent_graph.png")
    print("Agent graph rendered to agent_graph.png")

    # Run the agent
    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": "Calculate 2 + 2 * 3 and tell me the weather in New York",
            }
        ]
    }

    print("Running agent with tools...")
    print("Note: This will make real API calls to OpenAI")
    print("You can set OPENAI_API_KEY environment variable to test this")

    try:
        result = await agent.arun(initial_state)

        print("\nFinal messages:")
        for msg in result["messages"]:
            print(f"  {msg.type}: {msg.content}")

    except Exception as e:
        print(f"Error (likely missing API key): {e}")
        print("To test this, set your OPENAI_API_KEY environment variable")

    print("\n2️⃣ ReAct Tools with Scratchpad")
    print("-" * 35)

    # Build agent with ReAct tools and scratchpad
    react_agent = await (
        factory.with_chat(llm_config={"provider": "openai", "model": "gpt-4o-mini"})
        .with_react_tools([echo_tool], scratchpad_key="observations")
        .build()
    )

    # Render the ReAct agent's graph
    AgentFactory.diagram_agent(react_agent, "agent_graph_react.png")
    print("ReAct agent graph rendered to react_agent_graph.png")

    # Run the ReAct agent
    react_initial_state = {
        "messages": [
            {"role": "user", "content": "Echo the message 'Hello from Petal!'"}
        ]
    }

    print("Running ReAct agent with scratchpad...")

    try:
        react_result = await react_agent.arun(react_initial_state)

        print("Final messages:")
        for msg in react_result["messages"]:
            print(f"  {msg.type}: {msg.content}")

        if "observations" in react_result:
            print(f"Scratchpad observations: {react_result['observations']}")

    except Exception as e:
        print(f"Error (likely missing API key): {e}")
        print("To test this, set your OPENAI_API_KEY environment variable")

    print("\n3️⃣ Direct Tool Objects")
    print("-" * 25)

    # Show how to use tools directly without registration
    direct_agent = await (
        AgentFactory(ChatState)
        .with_chat(llm_config={"provider": "openai", "model": "gpt-4o-mini"})
        .with_tools([calculator])  # Direct tool object
        .build()
    )

    # Render the direct tool agent's graph
    AgentFactory.diagram_agent(direct_agent, "agent_graph_direct.png")
    print("Direct tool agent graph rendered to direct_agent_graph.png")

    print("Agent built with direct tool object ✓")

    print("\n4️⃣ Tool Registration and Resolution")
    print("-" * 35)

    # Show tool registry capabilities
    print("Registered tools:")
    for tool_name in registry.list():
        print(f"  • {tool_name}")

    print("\n✅ Tool Integration Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("  • Tool registration via ToolFactory")
    print("  • Tool resolution by string names")
    print("  • Direct tool object usage")
    print("  • ReAct-style tools with scratchpad")
    print("  • Full LLM → Tool → LLM loop")
    print("  • Tool message handling and state updates")
    print("\nTo test with real API calls:")
    print("  export OPENAI_API_KEY=your_api_key_here")
    print("  python playground3.py")


if __name__ == "__main__":
    asyncio.run(main())
