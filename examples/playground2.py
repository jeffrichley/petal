#!/usr/bin/env python3
"""
Interesting AgentFactory example demonstrating:
- User name input
- Random personality selection
- LLM greeting with personality-based system prompt
- Random topic selection
- LLM joke telling about the topic
"""

import asyncio
import random
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from petal.core.factory import AgentFactory
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Initialize Rich console
console = Console()


# Define the state type
class GreetingState(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str
    personality: str
    topic: str
    greeting: str
    joke: str


async def main():
    # Display title
    title = Panel(
        "[bold blue]🤖 AgentFactory: Greeting & Joke Agent[/bold blue]",
        box=box.ROUNDED,
        border_style="blue",
    )
    console.print(title)
    console.print()

    # Step 1: Select a random personality
    async def select_personality(_state: dict) -> dict:
        personalities = [
            "pirate",
            "robot",
            "wizard",
            "chef",
            "detective",
            "cowboy",
            "ninja",
            "scientist",
            "artist",
            "comedian",
        ]
        selected = random.choice(personalities)

        # Create a fancy panel for personality selection
        personality_panel = Panel(
            f"🎭 Selected personality: [bold green]{selected}[/bold green]",
            title="[bold]Personality Selection[/bold]",
            border_style="green",
            box=box.ROUNDED,
        )
        console.print(personality_panel)
        return {"personality": selected}

    # Step 2: Select a random topic for jokes
    async def select_topic(_state: dict) -> dict:
        topics = [
            "programming",
            "cooking",
            "travel",
            "pets",
            "weather",
            "technology",
            "food",
            "sports",
            "music",
            "books",
            "movies",
            "science",
            "history",
            "art",
            "nature",
        ]
        selected = random.choice(topics)

        # Create a fancy panel for topic selection
        topic_panel = Panel(
            f"🎯 Selected joke topic: [bold magenta]{selected}[/bold magenta]",
            title="[bold]Topic Selection[/bold]",
            border_style="magenta",
            box=box.ROUNDED,
        )
        console.print(topic_panel)
        return {"topic": selected}

    # Step 3: Generate a greeting with personality
    async def generate_greeting(state: dict) -> dict:
        greeting = f"Greeting generated for {state['user_name']} with {state['personality']} personality"

        # Create a fancy panel for greeting generation
        greeting_panel = Panel(
            Text(f"👋 {greeting}", style="blue"),
            title="[bold]Greeting Generation[/bold]",
            border_style="blue",
            box=box.ROUNDED,
        )
        console.print(greeting_panel)
        return {"greeting": greeting}

    # Step 4: Generate a joke about the topic
    async def generate_joke(state: dict) -> dict:
        joke = f"Joke generated about {state['topic']}"

        # Create a fancy panel for joke generation
        joke_panel = Panel(
            Text(f"😄 {joke}", style="yellow"),
            title="[bold]Joke Generation[/bold]",
            border_style="yellow",
            box=box.ROUNDED,
        )
        console.print(joke_panel)
        return {"joke": joke}

    # Show building progress
    with console.status("[bold green]Building the agent...", spinner="dots"):
        console.print()

    # Build the agent with AgentFactory
    agent = (
        AgentFactory(GreetingState)
        # Step 1: Select personality
        .add(select_personality)
        # Step 2: Generate greeting with personality-based system prompt
        .with_chat(
            prompt_template="Greet {user_name} in a friendly way!",
            system_prompt="You are a helpful assistant that speaks like a {personality}. Use your personality in your greeting!",
        )
        # Step 3: Select topic for joke
        .add(select_topic)
        # Step 4: Tell a joke about the topic
        .with_chat(
            prompt_template="Tell a funny joke about {topic}. Make it entertaining!",
            system_prompt="You are a {personality} who tells great jokes. Keep your personality in the joke!",
        ).build()
    )

    # Show running progress
    with console.status("[bold red]Running the agent...", spinner="dots"):
        console.print()

    # Run the agent with a user name
    user_name = "Jeff"

    # Create user info panel
    user_panel = Panel(
        f"👤 User: [bold blue]{user_name}[/bold blue]",
        title="[bold]User Information[/bold]",
        border_style="blue",
        box=box.ROUNDED,
    )
    console.print(user_panel)
    console.print()

    result = await agent.arun({"user_name": user_name, "messages": []})

    # Create results table
    results_table = Table(
        title="📋 Final Results", box=box.ROUNDED, border_style="green"
    )
    results_table.add_column("Field", style="cyan", no_wrap=True)
    results_table.add_column("Value", style="white")

    results_table.add_row("Personality", result.get("personality", "Unknown"))
    results_table.add_row("Topic", result.get("topic", "Unknown"))
    results_table.add_row("Greeting", result.get("greeting", "No greeting"))
    results_table.add_row("Joke", result.get("joke", "No joke"))

    console.print(results_table)
    console.print()

    # Show the conversation in a fancy way
    conversation_panel = Panel(
        "[bold magenta]💬 Agent Conversation[/bold magenta]",
        title="[bold]Conversation Log[/bold]",
        border_style="magenta",
        box=box.ROUNDED,
    )
    console.print(conversation_panel)

    for i, message in enumerate(result.get("messages", [])):
        role = (
            "🤖 AI"
            if hasattr(message, "content") and "AI" in str(type(message))
            else "👤 User"
        )
        content = message.content if hasattr(message, "content") else str(message)

        # Create message panel
        message_style = "green" if "AI" in role else "blue"
        message_panel = Panel(
            content,
            title=f"[bold]{role}[/bold]",
            border_style=message_style,
            box=box.ROUNDED,
        )
        console.print(message_panel)

        if i < len(result.get("messages", [])) - 1:
            console.print()


if __name__ == "__main__":
    asyncio.run(main())
