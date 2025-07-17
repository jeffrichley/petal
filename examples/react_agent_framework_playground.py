import asyncio
import logging

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from petal.core.factory import AgentFactory
from petal.core.tool_factory import ToolFactory
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

logging.getLogger("httpx").setLevel(logging.WARNING)

# Set up rich console
console = Console()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Define custom state schema ---
class CustomState(BaseModel):
    messages: list = Field(default_factory=list)
    user_name: str = "User"
    location: str = "Unknown"
    topic: str = "General"
    scratchpad: str = ""
    thoughts: list = Field(default_factory=list)
    actions: list = Field(default_factory=list)
    # Add any other fields you want in your final state
    weather_info: str = ""
    coffee_recommendations: str = ""


# --- Register tools with logging ---
tool_factory = ToolFactory()


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    console.print(f"[bold blue]🌤️  Calling get_weather with city: {city}[/bold blue]")
    result = f"It's always sunny in {city}!"
    console.print(f"[bold green]✅ Weather result: {result}[/bold green]")
    return result


tool_factory.add("get_weather", get_weather)

# --- Instantiate LLM ---
console.print("[bold yellow]🤖 Initializing LLM...[/bold yellow]")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# --- Build ReAct agent using AgentFactory ---
async def main():
    console.print(
        "[bold yellow]\U0001f527 Building ReAct agent with AgentFactory...[/bold yellow]"
    )
    agent = await (
        AgentFactory(CustomState)
        .with_react_loop(["get_weather"], llm_instance=llm, tool_factory=tool_factory)
        .with_system_prompt(
            "You are a helpful assistant. Address the user as {user_name}."
        )
        .with_structured_output(CustomState)
        .build()
    )
    console.print(
        "[bold green]\u2705 ReAct agent built successfully with AgentFactory![/bold green]"
    )

    # --- Run the agent ---
    console.print(
        "\n[bold cyan]\U0001f680 Starting ReAct Agent Framework Demo[/bold cyan]"
    )

    # Create initial state
    state = CustomState(
        messages=[
            HumanMessage(
                content="what is the weather in {location}? Also, tell me about {topic}."
            )
        ],
        user_name="Jeff",
        location="Chesapeake, VA",
        topic="the best coffee shops",
    )

    console.print(
        Panel(
            f"[bold]Initial State:[/bold]\n"
            f"User: {state.user_name}\n"
            f"Location: {state.location}\n"
            f"Topic: {state.topic}\n"
            f"Message: {state.messages[0].content}",
            title="\U0001f3af Input",
            border_style="blue",
        )
    )

    # Run the agent with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running ReAct loop...", total=None)

        console.print(
            "\n[bold magenta]\U0001f504 Starting ReAct reasoning loop...[/bold magenta]"
        )
        result = await agent.arun(state.model_dump())
        progress.update(task, completed=True)

    # Display final results
    console.print(
        "\n[bold green]\U0001f389 ReAct Agent Framework Completed![/bold green]"
    )

    # Create a results table
    table = Table(title="Final State Results")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("User Name", result["user_name"])
    table.add_row("Location", result["location"])
    table.add_row("Topic", result["topic"])
    table.add_row("Weather Info", result["weather_info"] or "Not populated")
    table.add_row(
        "Coffee Recommendations", result["coffee_recommendations"] or "Not populated"
    )
    table.add_row("Thoughts Count", str(len(result["thoughts"])))
    table.add_row("Actions Count", str(len(result["actions"])))

    console.print(table)

    # Show detailed scratchpad if available
    if result["scratchpad"]:
        console.print(
            Panel(
                result["scratchpad"],
                title="\U0001f4dd Reasoning Scratchpad",
                border_style="yellow",
            )
        )

    # Show thoughts and actions
    if result["thoughts"]:
        thoughts_text = "\n".join(
            [f"\U0001f4ad {thought}" for thought in result["thoughts"]]
        )
        console.print(
            Panel(thoughts_text, title="\U0001f9e0 Thoughts", border_style="blue")
        )

    if result["actions"]:
        actions_text = "\n".join([f"\u26a1 {action}" for action in result["actions"]])
        console.print(
            Panel(actions_text, title="\U0001f527 Actions", border_style="red")
        )


if __name__ == "__main__":
    console.print("[bold]\U0001f31f ReAct Agent Framework Playground[/bold]")
    console.print("=" * 50)
    asyncio.run(main())
