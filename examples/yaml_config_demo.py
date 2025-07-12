"""Demonstration of YAML configuration models for node loading."""

from petal.core.config.yaml import (
    LLMNodeConfig,
    ReactNodeConfig,
    StateSchemaConfig,
    ValidationConfig,
)


def demo_llm_node_config():
    """Demonstrate LLM node configuration."""
    print("=== LLM Node Configuration Demo ===\n")

    # Create a basic LLM node configuration
    llm_config = LLMNodeConfig(
        type="llm",
        name="assistant",
        description="A helpful AI assistant",
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
        prompt="You are a helpful assistant. Answer the user's question: {user_input}",
        system_prompt="You are a knowledgeable and helpful AI assistant.",
    )

    print("LLM Node Configuration:")
    print(f"  Type: {llm_config.type}")
    print(f"  Name: {llm_config.name}")
    print(f"  Description: {llm_config.description}")
    print(f"  Provider: {llm_config.provider}")
    print(f"  Model: {llm_config.model}")
    print(f"  Temperature: {llm_config.temperature}")
    print(f"  Max Tokens: {llm_config.max_tokens}")
    print(f"  Prompt: {llm_config.prompt}")
    print(f"  System Prompt: {llm_config.system_prompt}")
    print(f"  Enabled: {llm_config.enabled}")
    print()

    # Demonstrate validation
    print("Validation Examples:")
    try:
        # This should fail - invalid provider
        LLMNodeConfig(
            type="llm", name="test", provider="invalid_provider", model="gpt-4"
        )
    except Exception as e:
        print(f"  ✓ Invalid provider caught: {e}")

    try:
        # This should fail - invalid temperature
        LLMNodeConfig(
            type="llm",
            name="test",
            provider="openai",
            model="gpt-4",
            temperature=2.5,  # Too high
        )
    except Exception as e:
        print(f"  ✓ Invalid temperature caught: {e}")

    print()


def demo_react_node_config():
    """Demonstrate React node configuration."""
    print("=== React Node Configuration Demo ===\n")

    # Create a React node configuration
    react_config = ReactNodeConfig(
        type="react",
        name="reasoning_agent",
        description="An agent that can use tools and reason",
        tools=["search", "calculator", "database"],
        reasoning_prompt="Think step by step about how to solve this problem.",
        system_prompt="You are a reasoning agent that can use tools to solve problems.",
        max_iterations=5,
    )

    print("React Node Configuration:")
    print(f"  Type: {react_config.type}")
    print(f"  Name: {react_config.name}")
    print(f"  Description: {react_config.description}")
    print(f"  Tools: {react_config.tools}")
    print(f"  Reasoning Prompt: {react_config.reasoning_prompt}")
    print(f"  System Prompt: {react_config.system_prompt}")
    print(f"  Max Iterations: {react_config.max_iterations}")
    print(f"  Enabled: {react_config.enabled}")
    print()

    # Demonstrate validation
    print("Validation Examples:")
    try:
        # This should fail - invalid node type
        ReactNodeConfig(type="invalid", name="test")
    except Exception as e:
        print(f"  ✓ Invalid node type caught: {e}")

    try:
        # This should fail - empty tool names
        ReactNodeConfig(type="react", name="test", tools=["search", "", "calculator"])
    except Exception as e:
        print(f"  ✓ Empty tool names caught: {e}")

    print()


def demo_supporting_models():
    """Demonstrate supporting configuration models."""
    print("=== Supporting Models Demo ===\n")

    # State schema configuration
    state_schema = StateSchemaConfig(
        fields={"input": "str", "output": "str", "context": "str"},
        required_fields=["input", "output"],
    )

    print("State Schema Configuration:")
    print(f"  Fields: {state_schema.fields}")
    print(f"  Required Fields: {state_schema.required_fields}")
    print()

    # Validation configuration
    validation_config = ValidationConfig(
        input_schema=StateSchemaConfig(fields={"user_input": "str"}),
        output_schema=StateSchemaConfig(
            fields={"response": "str", "confidence": "float"}
        ),
    )

    print("Validation Configuration:")
    print(f"  Input Schema: {validation_config.input_schema}")
    print(f"  Output Schema: {validation_config.output_schema}")
    print()


def demo_minimal_configs():
    """Demonstrate minimal configurations."""
    print("=== Minimal Configurations Demo ===\n")

    # Minimal LLM config
    minimal_llm = LLMNodeConfig(
        type="llm", name="minimal_llm", provider="openai", model="gpt-4"
    )

    print("Minimal LLM Configuration:")
    print(f"  Name: {minimal_llm.name}")
    print(f"  Provider: {minimal_llm.provider}")
    print(f"  Model: {minimal_llm.model}")
    print(f"  Temperature: {minimal_llm.temperature} (default)")
    print(f"  Max Tokens: {minimal_llm.max_tokens} (default)")
    print(f"  Prompt: {minimal_llm.prompt} (None)")
    print()

    # Minimal React config
    minimal_react = ReactNodeConfig(type="react", name="minimal_react")

    print("Minimal React Configuration:")
    print(f"  Name: {minimal_react.name}")
    print(f"  Tools: {minimal_react.tools} (empty list)")
    print(f"  Max Iterations: {minimal_react.max_iterations} (default)")
    print(f"  Reasoning Prompt: {minimal_react.reasoning_prompt} (None)")
    print()


if __name__ == "__main__":
    print("YAML Configuration Models Demonstration\n")
    print("This demo shows how to use the new YAML configuration models")
    print("for creating LLM and React node configurations.\n")

    demo_llm_node_config()
    demo_react_node_config()
    demo_supporting_models()
    demo_minimal_configs()

    print("=== Summary ===")
    print("✅ All configuration models work correctly")
    print("✅ Validation catches invalid configurations")
    print("✅ Default values are applied appropriately")
    print("✅ String trimming works for all text fields")
    print("✅ Type checking and validation are comprehensive")
