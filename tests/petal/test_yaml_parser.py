import pytest

from petal.core.config.yaml import LLMNodeConfig
from petal.core.yaml.parser import YAMLFileNotFoundError, YAMLNodeParser, YAMLParseError


@pytest.fixture
def empty_yaml_file(tmp_path):
    file_path = tmp_path / "empty.yaml"
    file_path.write_text("")
    return str(file_path)


@pytest.fixture
def valid_llm_yaml_file(tmp_path):
    file_path = tmp_path / "llm_node.yaml"
    file_path.write_text(
        """
        type: llm
        name: assistant
        provider: openai
        model: gpt-4o-mini
        temperature: 0.0
        max_tokens: 1000
        prompt: "You are a helpful assistant."
        system_prompt: "You are a knowledgeable and helpful AI assistant."
        """
    )
    return str(file_path)


@pytest.fixture
def invalid_yaml_file(tmp_path):
    file_path = tmp_path / "invalid.yaml"
    file_path.write_text(
        """
        type: llm
        name: assistant
        provider: openai
        model: gpt-4o-mini
        temperature: 0.0
        max_tokens: 1000
        prompt: "You are a helpful assistant.
        system_prompt: "You are a knowledgeable and helpful AI assistant."
        """  # missing closing quote for prompt
    )
    return str(file_path)


def test_parse_node_config_empty_yaml(empty_yaml_file):
    parser = YAMLNodeParser()
    with pytest.raises(YAMLParseError, match="YAML root must be a mapping/object"):
        parser.parse_node_config(empty_yaml_file)


def test_parse_node_config_file_not_found():
    parser = YAMLNodeParser()
    with pytest.raises(YAMLFileNotFoundError, match="YAML file not found: "):
        parser.parse_node_config("this_file_does_not_exist.yaml")


def test_parse_node_config_invalid_yaml(invalid_yaml_file):
    parser = YAMLNodeParser()
    with pytest.raises(YAMLParseError):
        parser.parse_node_config(invalid_yaml_file)


def test_parse_node_config_returns_llm_node_config(valid_llm_yaml_file):
    parser = YAMLNodeParser()
    config = parser.parse_node_config(valid_llm_yaml_file)
    assert isinstance(config, LLMNodeConfig)
    assert config.type == "llm"
    assert config.name == "assistant"
    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.0
    assert config.max_tokens == 1000
    assert config.prompt == "You are a helpful assistant."
    assert config.system_prompt == "You are a knowledgeable and helpful AI assistant."


def test_parse_node_config_unsupported_node_type(tmp_path):
    """Test parsing YAML with unsupported node type."""
    file_path = tmp_path / "unsupported.yaml"
    file_path.write_text(
        """
        type: unsupported
        name: test
        """
    )
    parser = YAMLNodeParser()
    with pytest.raises(ValueError, match="Unsupported node type: unsupported"):
        parser.parse_node_config(str(file_path))


def test_parse_node_config_missing_type(tmp_path):
    """Test parsing YAML with missing type field."""
    file_path = tmp_path / "missing_type.yaml"
    file_path.write_text(
        """
        name: test
        provider: openai
        model: gpt-4
        """
    )
    parser = YAMLNodeParser()
    with pytest.raises(ValueError, match="Unsupported node type: None"):
        parser.parse_node_config(str(file_path))


def test_load_yaml_file_not_implemented():
    """Test that load_yaml_file raises NotImplementedError."""
    parser = YAMLNodeParser()
    with pytest.raises(NotImplementedError):
        parser.load_yaml_file("test.yaml")


def test_detect_node_type_not_implemented():
    """Test that detect_node_type raises NotImplementedError."""
    parser = YAMLNodeParser()
    with pytest.raises(NotImplementedError):
        parser.detect_node_type({})


def test_validate_yaml_schema_not_implemented():
    """Test that validate_yaml_schema raises NotImplementedError."""
    parser = YAMLNodeParser()
    with pytest.raises(NotImplementedError):
        parser.validate_yaml_schema({})
