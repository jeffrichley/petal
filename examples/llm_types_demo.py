"""Demonstration of LLMTypes preconfigured configurations with AgentFactory."""

import asyncio

from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from petal.core import AgentFactory, LLMTypes


# Define state types
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    analysis_result: str
    creative_response: str


async def main():
    """Demonstrate different LLM configurations using LLMTypes."""

    print("🤖 LLMTypes Demo: Multi-LLM Agent")
    print("=" * 50)

    # Create an agent with multiple LLM steps using different configurations
    agent = (
        AgentFactory(ChatState)
        # Step 1: Analytical analysis with GPT-4o-mini
        .with_chat(
            llm_config=LLMTypes.OPENAI_GPT4O_MINI_ANALYTICAL,
            prompt_template="Analyze this input: {user_input}",
            system_prompt="You are an analytical assistant. Provide clear, logical analysis.",
        )
        # Step 2: Creative response with GPT-4o (creative)
        .with_chat(
            llm_config=LLMTypes.OPENAI_GPT4O_CREATIVE,
            prompt_template="Based on the analysis, create a creative response about: {user_input}",
            system_prompt="You are a creative writer. Be imaginative and engaging.",
        ).build()
    )

    # Test the agent
    user_input = "artificial intelligence and its impact on society"

    print(f"📝 User Input: {user_input}")
    print("\n🔄 Running agent with multiple LLM configurations...")

    result = await agent.arun({"user_input": user_input, "messages": []})

    print("\n📊 Results:")
    print(f"Analysis: {result.get('analysis_result', 'No analysis')}")
    print(
        f"Creative Response: {result.get('creative_response', 'No creative response')}"
    )

    print("\n💬 Conversation:")
    for i, message in enumerate(result.get("messages", [])):
        role = "🤖 AI" if hasattr(message, "content") else "👤 User"
        content = message.content if hasattr(message, "content") else str(message)
        print(f"{role}: {content}")
        if i < len(result.get("messages", [])) - 1:
            print()


async def demo_custom_configurations():
    """Demonstrate custom LLM configurations."""

    print("\n" + "=" * 50)
    print("🔧 Custom LLM Configurations Demo")
    print("=" * 50)

    # Create custom configurations
    custom_analytical = LLMTypes.create_custom(
        "openai", "gpt-4o", temperature=0.1, max_tokens=8000
    )

    custom_creative = LLMTypes.with_temperature(LLMTypes.OPENAI_GPT4O, temperature=0.9)

    custom_long_response = LLMTypes.with_max_tokens(
        LLMTypes.OPENAI_GPT4O_LARGE, max_tokens=8000
    )

    print("Custom Configurations Created:")
    print(f"📊 Analytical: {custom_analytical}")
    print(f"🎨 Creative: {custom_creative}")
    print(f"📝 Long Response: {custom_long_response}")

    # Test with custom configuration
    agent = (
        AgentFactory(ChatState)
        .with_chat(
            llm_config=custom_analytical,
            prompt_template="Provide detailed analysis of: {user_input}",
            system_prompt="You are a detailed analyst.",
        )
        .build()
    )

    await agent.arun({"user_input": "climate change solutions", "messages": []})

    print("\n✅ Custom configuration test completed!")


async def demo_provider_variety():
    """Demonstrate different provider configurations."""

    print("\n" + "=" * 50)
    print("🌐 Provider Variety Demo")
    print("=" * 50)

    # Show available configurations for different providers
    providers = {
        "OpenAI": [
            LLMTypes.OPENAI_GPT4O_MINI,
            LLMTypes.OPENAI_GPT4O,
            LLMTypes.OPENAI_GPT4O_LARGE,
            LLMTypes.OPENAI_GPT35_TURBO,
        ],
        "Anthropic": [
            LLMTypes.ANTHROPIC_CLAUDE_3_HAIKU,
            LLMTypes.ANTHROPIC_CLAUDE_3_SONNET,
            LLMTypes.ANTHROPIC_CLAUDE_3_OPUS,
        ],
        "Google": [
            LLMTypes.GOOGLE_GEMINI_PRO,
            LLMTypes.GOOGLE_GEMINI_PRO_CREATIVE,
        ],
        "Cohere": [
            LLMTypes.COHERE_COMMAND,
            LLMTypes.COHERE_COMMAND_CREATIVE,
        ],
    }

    for provider, configs in providers.items():
        print(f"\n{provider} Configurations:")
        for config in configs:
            print(f"  • {config.model} (temp: {config.temperature})")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(demo_custom_configurations())
    asyncio.run(demo_provider_variety())
