from unittest.mock import patch

import pytest
from tests.petal.conftest_factory import DefaultState

from petal.core.config.yaml import LLMNodeConfig, ReactNodeConfig
from petal.core.factory import AgentFactory
from petal.core.yaml.parser import YAMLFileNotFoundError


@pytest.mark.asyncio
async def test_agent_factory_node_from_yaml_llm():
    """Test AgentFactory.node_from_yaml loads LLM node from YAML and adds to builder."""
    factory = AgentFactory(DefaultState)
    mock_config = LLMNodeConfig(
        type="llm",
        name="test_llm",
        description="Test LLM node",
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
        prompt="Test prompt",
        system_prompt="Test system",
    )
    with (
        patch("petal.core.yaml.parser.YAMLNodeParser") as mock_parser,
        patch("petal.core.yaml.handlers.HandlerFactory") as mock_factory,
    ):
        mock_parser.return_value.parse_node_config.return_value = mock_config
        mock_handler = patch("petal.core.yaml.handlers.llm.LLMNodeHandler").start()
        mock_factory.return_value.get_handler.return_value = mock_handler

        async def mock_create_node(config):  # noqa: ARG001
            return lambda x: x

        mock_handler.create_node = mock_create_node

        # Should add to builder and return function
        node = await factory.node_from_yaml("test.yaml")
        assert callable(node)

        # Should have added a step to the builder
        assert len(factory._builder._config.steps) == 1
        assert factory._builder._config.steps[0].strategy_type == "custom"
        assert factory._builder._config.steps[0].node_name == "test_llm"

        patch.stopall()


@pytest.mark.asyncio
async def test_agent_factory_node_from_yaml_react():
    """Test AgentFactory.node_from_yaml loads React node from YAML and adds to builder."""
    factory = AgentFactory(DefaultState)
    mock_config = ReactNodeConfig(
        type="react",
        name="test_react",
        description="Test React node",
        tools=["search", "calculator"],
        reasoning_prompt="Think step by step",
        system_prompt="You are a reasoning agent",
        max_iterations=5,
    )
    with (
        patch("petal.core.yaml.parser.YAMLNodeParser") as mock_parser,
        patch("petal.core.yaml.handlers.HandlerFactory") as mock_factory,
    ):
        mock_parser.return_value.parse_node_config.return_value = mock_config
        mock_handler = patch("petal.core.yaml.handlers.react.ReactNodeHandler").start()
        mock_factory.return_value.get_handler.return_value = mock_handler

        async def mock_create_node(config):  # noqa: ARG001
            return lambda x: x

        mock_handler.create_node = mock_create_node

        # Should add to builder and return function
        node = await factory.node_from_yaml("test.yaml")
        assert callable(node)

        # Should have added a step to the builder
        assert len(factory._builder._config.steps) == 1
        assert factory._builder._config.steps[0].strategy_type == "custom"
        assert factory._builder._config.steps[0].node_name == "test_react"

        patch.stopall()


@pytest.mark.asyncio
async def test_agent_factory_node_from_yaml_file_not_found():
    """Test AgentFactory.node_from_yaml handles file not found."""
    factory = AgentFactory(DefaultState)
    with pytest.raises(YAMLFileNotFoundError):
        await factory.node_from_yaml("nonexistent.yaml")


@pytest.mark.asyncio
async def test_agent_factory_node_from_yaml_invalid_yaml():
    """Test that node_from_yaml raises YAMLParseError for invalid YAML."""
    factory = AgentFactory(DefaultState)
    with pytest.raises(YAMLFileNotFoundError):
        await factory.node_from_yaml("nonexistent_file.yaml")
