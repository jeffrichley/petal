import pytest
from pydantic import ValidationError

from petal.core.config.yaml import (
    CustomNodeConfig,
    LLMNodeConfig,
    ReactNodeConfig,
    ToolDiscoveryConfig,
)


@pytest.mark.asyncio
async def test_react_node_yaml_zero_config_tool_discovery():
    """React nodes use default tool discovery when no config provided."""
    # Arrange
    config = ReactNodeConfig(
        type="react",
        name="reasoning_agent",
        tools=["search", "calculator"],
        reasoning_prompt="Think step by step",
    )
    # Act & Assert
    # Should be None for zero-config; AgentFactory should use defaults
    assert config.tool_discovery is None


@pytest.mark.asyncio
async def test_react_node_yaml_partial_tool_discovery_override():
    """Partial tool discovery config overrides only specified fields."""
    # Arrange
    config = ReactNodeConfig(
        type="react",
        name="reasoning_agent",
        tools=["search", "calculator"],
        tool_discovery=ToolDiscoveryConfig(
            folders=["my_tools/"], exclude_patterns=["temp_*"]
        ),
        reasoning_prompt="Think step by step",
    )
    # Act & Assert
    assert config.tool_discovery is not None
    assert config.tool_discovery.folders == ["my_tools/"]
    assert config.tool_discovery.exclude_patterns == ["temp_*"]
    # enabled and config_locations should use defaults (None or True)


@pytest.mark.asyncio
async def test_react_node_yaml_tool_discovery_disabled():
    """Tool discovery can be disabled in YAML config."""
    # Arrange
    config = ReactNodeConfig(
        type="react",
        name="reasoning_agent",
        tools=["search", "calculator"],
        tool_discovery=ToolDiscoveryConfig(enabled=False),
        reasoning_prompt="Think step by step",
    )
    # Act & Assert
    assert config.tool_discovery is not None
    assert config.tool_discovery.enabled is False


def test_react_node_yaml_invalid_tool_discovery_config():
    """Validation of invalid tool discovery configuration (bad type)."""
    # Arrange & Act & Assert
    with pytest.raises(ValidationError, match="Input should be a valid boolean"):
        ReactNodeConfig(
            type="react",
            name="reasoning_agent",
            tools=["search", "calculator"],
            tool_discovery=ToolDiscoveryConfig(
                enabled="not_a_boolean",  # type: ignore[arg-type]  # Invalid type
            ),
            reasoning_prompt="Think step by step",
        )


@pytest.mark.asyncio
async def test_react_node_yaml_full_tool_discovery_config():
    """Full tool discovery configuration override."""
    # Arrange
    config = ReactNodeConfig(
        type="react",
        name="reasoning_agent",
        tools=["search", "calculator"],
        tool_discovery=ToolDiscoveryConfig(
            enabled=True,
            folders=["my_tools/"],
            config_locations=["my_configs/"],
            exclude_patterns=["temp_*"],
        ),
        reasoning_prompt="Think step by step",
    )
    # Act & Assert
    assert config.tool_discovery is not None
    assert config.tool_discovery.enabled is True
    assert config.tool_discovery.folders == ["my_tools/"]
    assert config.tool_discovery.config_locations == ["my_configs/"]
    assert config.tool_discovery.exclude_patterns == ["temp_*"]


def test_react_node_yaml_invalid_folders_type():
    """Validation of invalid folders type in tool discovery."""
    # Arrange & Act & Assert
    with pytest.raises(ValidationError, match="Input should be a valid list"):
        ReactNodeConfig(
            type="react",
            name="reasoning_agent",
            tools=["search", "calculator"],
            tool_discovery=ToolDiscoveryConfig(
                folders="not_a_list",  # type: ignore[arg-type]  # Invalid type
            ),
            reasoning_prompt="Think step by step",
        )


# LLM Node Tests
@pytest.mark.asyncio
async def test_llm_node_yaml_zero_config_tool_discovery():
    """LLM nodes use default tool discovery when no config provided."""
    # Arrange
    config = LLMNodeConfig(
        type="llm",
        name="assistant",
        provider="openai",
        model="gpt-4i",
        prompt="You are a helpful assistant",
    )
    # Act & Assert
    assert config.tool_discovery is None


@pytest.mark.asyncio
async def test_llm_node_yaml_tool_discovery_override():
    """LLM nodes can override tool discovery config."""
    # Arrange
    config = LLMNodeConfig(
        type="llm",
        name="assistant",
        provider="openai",
        model="gpt-4i",
        prompt="You are a helpful assistant",
        tool_discovery=ToolDiscoveryConfig(enabled=False, folders=["llm_tools/"]),
    )
    # Act & Assert
    assert config.tool_discovery is not None
    assert config.tool_discovery.enabled is False
    assert config.tool_discovery.folders == ["llm_tools/"]


# Custom Node Tests
@pytest.mark.asyncio
async def test_custom_node_yaml_zero_config_tool_discovery():
    """Custom nodes use default tool discovery when no config provided."""
    # Arrange
    config = CustomNodeConfig(
        type="custom",
        name="data_processor",
        function_path="my_module.process_data",
    )
    # Act & Assert
    assert config.tool_discovery is None


@pytest.mark.asyncio
async def test_custom_node_yaml_tool_discovery_override():
    """Custom nodes can override tool discovery config."""
    # Arrange
    config = CustomNodeConfig(
        type="custom",
        name="data_processor",
        function_path="my_module.process_data",
        tool_discovery=ToolDiscoveryConfig(
            enabled=True, config_locations=["custom_configs/"]
        ),
    )
    # Act & Assert
    assert config.tool_discovery is not None
    assert config.tool_discovery.enabled is True
    assert config.tool_discovery.config_locations == ["custom_configs/"]


# Edge Case Tests
def test_tool_discovery_empty_lists():
    """Discovery accepts empty lists for folders and config_locations."""
    # Arrange
    config = ReactNodeConfig(
        type="react",
        name="reasoning_agent",
        tools=["search", "calculator"],
        tool_discovery=ToolDiscoveryConfig(
            folders=[], config_locations=[], exclude_patterns=[]
        ),
        reasoning_prompt="Think step by step",
    )
    # Act & Assert
    assert config.tool_discovery is not None
    assert config.tool_discovery.folders == []
    assert config.tool_discovery.config_locations == []
    assert config.tool_discovery.exclude_patterns == []


def test_tool_discovery_none_values():
    """Discovery accepts None values for optional fields."""
    # Arrange
    config = ReactNodeConfig(
        type="react",
        name="reasoning_agent",
        tools=["search", "calculator"],
        tool_discovery=ToolDiscoveryConfig(
            enabled=True, folders=None, config_locations=None, exclude_patterns=None
        ),
        reasoning_prompt="Think step by step",
    )
    # Act & Assert
    assert config.tool_discovery is not None
    assert config.tool_discovery.enabled is True
    assert config.tool_discovery.folders is None
    assert config.tool_discovery.config_locations is None
    assert config.tool_discovery.exclude_patterns is None


def test_tool_discovery_invalid_config_locations_type():
    """Validation of invalid config_locations type in tool discovery."""
    # Arrange & Act & Assert
    with pytest.raises(ValidationError, match="Input should be a valid list"):
        ReactNodeConfig(
            type="react",
            name="reasoning_agent",
            tools=["search", "calculator"],
            tool_discovery=ToolDiscoveryConfig(
                config_locations="not_a_list",  # type: ignore[arg-type]  # Invalid type
            ),
            reasoning_prompt="Think step by step",
        )


def test_tool_discovery_invalid_exclude_patterns_type():
    """Validation of invalid exclude_patterns type in tool discovery."""
    # Arrange & Act & Assert
    with pytest.raises(ValidationError, match="Input should be a valid list"):
        ReactNodeConfig(
            type="react",
            name="reasoning_agent",
            tools=["search", "calculator"],
            tool_discovery=ToolDiscoveryConfig(
                exclude_patterns="not_a_list",  # type: ignore[arg-type]  # Invalid type
            ),
            reasoning_prompt="Think step by step",
        )


# Integration Tests
@pytest.mark.asyncio
async def test_yaml_parser_with_tool_discovery():
    """ML parser can load configs with tool discovery."""
    import os
    import tempfile

    from petal.core.yaml.parser import YAMLNodeParser

    # Arrange
    yaml_content = """
type: react
name: reasoning_agent
tools: [search, calculator]
tool_discovery:
  enabled: true
  folders: [my_tools/]
  config_locations: [my_configs/]
  exclude_patterns: [temp_*]
reasoning_prompt: "Think step by step"
"""
    # Act
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        parser = YAMLNodeParser()
        config = parser.parse_node_config(temp_path)

        # Assert
        assert config.tool_discovery is not None
        assert config.tool_discovery.enabled is True
        assert config.tool_discovery.folders == ["my_tools/"]
        assert config.tool_discovery.config_locations == ["my_configs/"]
        assert config.tool_discovery.exclude_patterns == ["temp_*"]
    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_yaml_parser_without_tool_discovery():
    """YAML parser works without tool discovery config."""
    import os
    import tempfile

    from petal.core.yaml.parser import YAMLNodeParser

    # Arrange
    yaml_content = """
type: react
name: reasoning_agent
tools: [search, calculator]
reasoning_prompt: "Think step by step"
"""
    # Act
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        parser = YAMLNodeParser()
        config = parser.parse_node_config(temp_path)

        # Assert
        assert config.tool_discovery is None
    finally:
        os.unlink(temp_path)
