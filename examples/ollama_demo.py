"""
Ollama Integration Demo

This example demonstrates how to use local LLMs via Ollama with the Petal framework.
Make sure you have Ollama installed and running locally with the required models.

Installation:
1. Install Ollama: https://ollama.ai/
2. Pull a model: ollama pull qwen3:4b
3. Start Ollama service: ollama serve

Usage:
    python examples/ollama_demo.py
"""

import asyncio
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from petal.core.factory import AgentFactory


class OllamaState(TypedDict):
    """State for Ollama agent example."""

    messages: Annotated[list, add_messages]
    user_input: str
    model_name: str


async def main():
    """Demonstrate Ollama integration with Petal using qwen3:4b."""

    # Example 1: Simple Ollama agent with qwen3:4b
    print("🤖 Creating Ollama agent with qwen3:4b...")

    agent = await (
        AgentFactory(OllamaState)
        .with_chat(
            prompt_template="User says: {user_input}. Respond naturally.",
            system_prompt="You are a helpful AI assistant running locally via Ollama (qwen3:4b).",
            llm_config={
                "provider": "ollama",
                "model": "qwen3:4b",
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        )
        .build()
    )

    result = await agent.arun(
        {
            "user_input": "Hello! What's the weather like today?",
            "model_name": "qwen3:4b",
            "messages": [],
        }
    )

    print(f"📝 Response: {result['messages'][-1].content}")
    print()

    # Example 2: Creative agent with qwen3:4b
    print("🎨 Creating creative agent with qwen3:4b...")

    creative_agent = await (
        AgentFactory(OllamaState)
        .with_chat(
            prompt_template="User asks: {user_input}. Be creative and engaging in your response.",
            system_prompt="You are a creative and imaginative AI assistant. Be playful and entertaining (qwen3:4b).",
            llm_config={
                "provider": "ollama",
                "model": "qwen3:4b",
                "temperature": 0.9,
                "max_tokens": 800,
            },
        )
        .build()
    )

    result = await creative_agent.arun(
        {
            "user_input": "Tell me a short story about a robot learning to paint",
            "model_name": "qwen3:4b",
            "messages": [],
        }
    )

    print(f"🎭 Creative Response: {result['messages'][-1].content}")
    print()

    # Example 3: Using preconfigured LLM types (manually override to qwen3:4b)
    print("⚙️ Using preconfigured Ollama configuration (qwen3:4b)...")

    from petal.core.config.llm_types import LLMTypes

    analytical_agent = await (
        AgentFactory(OllamaState)
        .with_chat(
            prompt_template="Analyze this: {user_input}. Provide a detailed, analytical response.",
            system_prompt="You are an analytical AI assistant. Be thorough and precise (qwen3:4b).",
            llm_config={**LLMTypes.OLLAMA_LLAMA2.model_dump(), "model": "qwen3:4b"},
        )
        .build()
    )

    result = await analytical_agent.arun(
        {
            "user_input": "What are the benefits of using local LLMs?",
            "model_name": "qwen3:4b",
            "messages": [],
        }
    )

    print(f"🔍 Analytical Response: {result['messages'][-1].content}")
    print()

    # Example 4: Custom Ollama configuration (qwen3:4b)
    print("🔧 Using custom Ollama configuration (qwen3:4b)...")

    custom_agent = await (
        AgentFactory(OllamaState)
        .with_chat(
            prompt_template="Code this: {user_input}. Provide clean, well-commented code.",
            system_prompt="You are a coding assistant. Write clean, efficient code with good comments (qwen3:4b).",
            llm_config={
                "provider": "ollama",
                "model": "qwen3:4b",
                "temperature": 0.1,  # Low temperature for consistent code
                "max_tokens": 2000,
                "base_url": "http://localhost:11434",  # Explicit base URL
            },
        )
        .build()
    )

    result = await custom_agent.arun(
        {
            "user_input": "Write a Python function to calculate fibonacci numbers",
            "model_name": "qwen3:4b",
            "messages": [],
        }
    )

    print(f"💻 Code Response: {result['messages'][-1].content}")


if __name__ == "__main__":
    print("🚀 Ollama Integration Demo (qwen3:4b)")
    print("=" * 50)
    print("Make sure Ollama is running: ollama serve")
    print("And you have the required model: ollama pull qwen3:4b")
    print("=" * 50)
    print()

    asyncio.run(main())
