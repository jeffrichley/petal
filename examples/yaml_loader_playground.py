#!/usr/bin/env python3
"""
YAML Loader Playground

Demonstrates the YAML parser functionality for loading node configurations.
This script shows how to use the YAMLNodeParser to load LLM nodes from YAML files.
"""

import os
import tempfile

from petal.core.config.yaml import LLMNodeConfig
from petal.core.yaml.parser import YAMLFileNotFoundError, YAMLNodeParser, YAMLParseError


def create_test_yaml_files():
    """Create temporary YAML files for testing."""
    files = {}

    # Valid LLM node configuration
    llm_config = """
type: llm
name: assistant
description: A helpful AI assistant
provider: openai
model: gpt-4o-mini
temperature: 0.0
max_tokens: 1000
prompt: "You are a helpful assistant. Answer the user's question: {user_input}"
system_prompt: "You are a knowledgeable and helpful AI assistant."
"""

    # Invalid YAML (missing closing quote)
    invalid_config = """
type: llm
name: assistant
provider: openai
model: gpt-4o-mini
temperature: 0.0
max_tokens: 1000
prompt: "You are a helpful assistant.
system_prompt: "You are a knowledgeable and helpful AI assistant."
"""

    # Empty file
    empty_config = ""

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(llm_config)
        files["valid_llm"] = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(invalid_config)
        files["invalid_yaml"] = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(empty_config)
        files["empty"] = f.name

    return files


def test_valid_llm_config(parser, files):
    """Test loading valid LLM configuration."""
    print("\n✅ Test 1: Loading Valid LLM Configuration")
    print("-" * 40)
    try:
        config = parser.parse_node_config(files["valid_llm"])
        print("✅ Successfully loaded LLM node configuration:")
        print(f"  Type: {config.type}")
        print(f"  Name: {config.name}")
        print(f"  Description: {config.description}")
        print(f"  Provider: {config.provider}")
        print(f"  Model: {config.model}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Max Tokens: {config.max_tokens}")
        print(f"  Prompt: {config.prompt}")
        print(f"  System Prompt: {config.system_prompt}")
        print(f"  Enabled: {config.enabled}")

        # Verify it's the correct type
        assert isinstance(config, LLMNodeConfig)
        print("✅ Confirmed: config is an instance of LLMNodeConfig")

    except Exception as e:
        print(f"❌ Error loading valid LLM config: {e}")


def test_invalid_yaml(parser, files):
    """Test loading invalid YAML syntax."""
    print("\n❌ Test 2: Loading Invalid YAML")
    print("-" * 40)
    try:
        config = parser.parse_node_config(files["invalid_yaml"])
        print(f"❌ Unexpected success: {config}")
    except YAMLParseError as e:
        print(f"✅ Correctly caught YAML parse error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")


def test_empty_yaml(parser, files):
    """Test loading empty YAML file."""
    print("\n❌ Test 3: Loading Empty YAML File")
    print("-" * 40)
    try:
        config = parser.parse_node_config(files["empty"])
        print(f"❌ Unexpected success: {config}")
    except YAMLParseError as e:
        print(f"✅ Correctly caught YAML parse error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")


def test_nonexistent_file(parser):
    """Test loading non-existent file."""
    print("\n❌ Test 4: Loading Non-existent File")
    print("-" * 40)
    try:
        config = parser.parse_node_config("this_file_does_not_exist.yaml")
        print(f"❌ Unexpected success: {config}")
    except YAMLFileNotFoundError as e:
        print(f"✅ Correctly caught file not found error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")


def test_unsupported_node_type(parser):
    """Test loading unsupported node type."""
    print("\n❌ Test 5: Loading Unsupported Node Type")
    print("-" * 40)

    # Create a file with unsupported type
    unsupported_config = """
type: custom
name: unsupported
function_path: "my_module.function"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(unsupported_config)
        unsupported_file = f.name

    try:
        config = parser.parse_node_config(unsupported_file)
        print(f"❌ Unexpected success: {config}")
    except ValueError as e:
        print(f"✅ Correctly caught unsupported type error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")

    return unsupported_file


def cleanup_files(files, unsupported_file=None):
    """Clean up temporary test files."""
    print("\n🧹 Cleanup")
    print("-" * 40)
    for name, path in files.items():
        try:
            os.unlink(path)
            print(f"✅ Deleted {name}: {path}")
        except Exception as e:
            print(f"❌ Failed to delete {name}: {e}")

    if unsupported_file:
        try:
            os.unlink(unsupported_file)
            print(f"✅ Deleted unsupported type file: {unsupported_file}")
        except Exception as e:
            print(f"❌ Failed to delete unsupported type file: {e}")


def demo_yaml_parser():
    """Demonstrate YAML parser functionality."""
    print("🎯 YAML Loader Playground")
    print("=" * 50)

    # Create test files
    files = create_test_yaml_files()
    parser = YAMLNodeParser()

    print("\n📁 Test Files Created:")
    for name, path in files.items():
        print(f"  {name}: {path}")

    # Run all tests
    test_valid_llm_config(parser, files)
    test_invalid_yaml(parser, files)
    test_empty_yaml(parser, files)
    test_nonexistent_file(parser)
    unsupported_file = test_unsupported_node_type(parser)

    # Cleanup
    cleanup_files(files, unsupported_file)

    print("\n🎉 YAML Loader Playground Complete!")
    print("=" * 50)


if __name__ == "__main__":
    demo_yaml_parser()
