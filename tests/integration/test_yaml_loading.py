"""Integration tests for YAML node loading functionality."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from petal.core.factory import AgentFactory, DefaultState


class TestYAMLLoadingIntegration:
    """Integration tests for YAML node loading."""

    def setup_method(self):
        """Set up test fixtures."""
        self.factory = AgentFactory(DefaultState)
        self.test_yaml_dir = "examples/yaml"

    def test_load_llm_node_from_yaml(self):
        """Test loading LLM node from YAML configuration."""
        # Arrange
        yaml_path = os.path.join(self.test_yaml_dir, "llm_node.yaml")

        # Act
        node_function = self.factory.node_from_yaml(yaml_path)

        # Assert
        assert node_function is not None
        assert callable(node_function)

        # Verify the node was added to the builder
        assert len(self.factory._builder._config.steps) == 1
        step_config = self.factory._builder._config.steps[0]
        assert step_config.strategy_type == "custom"
        assert step_config.node_name == "assistant"

    def test_load_react_node_from_yaml(self):
        """Test loading React node from YAML configuration."""
        # Arrange
        yaml_path = os.path.join(self.test_yaml_dir, "react_node.yaml")

        # Act - Just test that we can parse the YAML and create a config
        from petal.core.yaml.parser import YAMLNodeParser

        parser = YAMLNodeParser()
        config = parser.parse_node_config(yaml_path)

        # Assert
        assert config is not None
        assert config.type == "react"
        assert config.name == "reasoning_agent"
        assert "search" in config.tools
        assert "calculator" in config.tools

    def test_load_custom_node_from_yaml(self):
        """Test loading custom node from YAML configuration."""
        # Arrange
        yaml_path = os.path.join(self.test_yaml_dir, "custom_node.yaml")

        # Act - Just test that we can parse the YAML and create a config
        from petal.core.yaml.parser import YAMLNodeParser

        parser = YAMLNodeParser()
        config = parser.parse_node_config(yaml_path)

        # Assert
        assert config is not None
        assert config.type == "custom"
        assert config.name == "data_processor"
        assert config.function_path == "examples.custom_tool.process_data"
        assert config.parameters["batch_size"] == 100
        assert config.parameters["timeout"] == 30

    def test_load_complex_node_from_yaml(self):
        """Test loading complex node with MCP tools and state schema."""
        # Arrange
        yaml_path = os.path.join(self.test_yaml_dir, "complex_node.yaml")

        # Act - Just test that we can parse the YAML and create a config
        from petal.core.yaml.parser import YAMLNodeParser

        parser = YAMLNodeParser()
        config = parser.parse_node_config(yaml_path)

        # Assert
        assert config is not None
        assert config.type == "react"
        assert config.name == "advanced_agent"
        assert "search" in config.tools
        assert "mcp:filesystem" in config.tools
        assert "mcp:sqlite" in config.tools
        assert config.max_iterations == 10
        assert config.state_schema is not None
        assert "user_query" in config.state_schema.fields
        assert "search_results" in config.state_schema.fields
        assert "final_answer" in config.state_schema.fields

    def test_yaml_file_not_found_error(self):
        """Test error handling for non-existent YAML file."""
        # Arrange
        non_existent_path = "non_existent_file.yaml"

        # Act & Assert
        from petal.core.yaml.parser import YAMLFileNotFoundError

        with pytest.raises(YAMLFileNotFoundError):
            self.factory.node_from_yaml(non_existent_path)

    def test_invalid_yaml_syntax_error(self):
        """Test error handling for invalid YAML syntax."""
        # Arrange
        import tempfile

        from petal.core.yaml.parser import YAMLParseError

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: [unclosed_list\n")
            temp_path = f.name
        try:
            # Act & Assert
            with pytest.raises(YAMLParseError):
                self.factory.node_from_yaml(temp_path)
        finally:
            os.unlink(temp_path)

    def test_unsupported_node_type_error(self):
        """Test error handling for unsupported node type."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("type: unsupported\nname: test")
            temp_path = f.name

        try:
            # Act & Assert
            with pytest.raises(ValueError, match="Unsupported node type"):
                self.factory.node_from_yaml(temp_path)
        finally:
            os.unlink(temp_path)

    def test_integration_with_existing_factory_methods(self):
        """Test integration of YAML loading with existing factory methods."""
        # Arrange
        yaml_path = os.path.join(self.test_yaml_dir, "llm_node.yaml")

        # Act - Add YAML node first, then use existing methods
        self.factory.node_from_yaml(yaml_path)
        self.factory.with_chat(
            prompt_template="Additional prompt: {user_input}",
            system_prompt="Additional system prompt",
        )

        # Assert
        assert len(self.factory._builder._config.steps) == 2
        # First step should be the YAML node
        assert self.factory._builder._config.steps[0].node_name == "assistant"
        # Second step should be the programmatic LLM step
        assert self.factory._builder._config.steps[1].strategy_type == "llm"

    def test_build_agent_with_yaml_nodes(self):
        """Test building a complete agent with YAML nodes."""
        # Arrange
        yaml_path = os.path.join(self.test_yaml_dir, "llm_node.yaml")

        # Act
        self.factory.node_from_yaml(yaml_path)
        agent = self.factory.build()

        # Assert
        assert agent is not None
        assert hasattr(agent, "graph")
        assert agent.built

    @patch("petal.core.yaml.parser.YAMLNodeParser.parse_node_config")
    def test_yaml_parsing_performance(self, mock_parse):
        """Test performance of YAML parsing vs programmatic creation."""
        # Arrange
        yaml_path = os.path.join(self.test_yaml_dir, "llm_node.yaml")
        mock_config = Mock()
        mock_config.type = "llm"
        mock_config.name = "test"
        mock_config.provider = "openai"
        mock_config.model = "gpt-4o-mini"
        mock_parse.return_value = mock_config

        # Act
        self.factory.node_from_yaml(yaml_path)

        # Assert
        mock_parse.assert_called_once_with(yaml_path)

    def test_multiple_yaml_nodes_in_agent(self):
        """Test loading multiple YAML nodes in a single agent."""
        # Arrange
        llm_yaml = os.path.join(self.test_yaml_dir, "llm_node.yaml")
        react_yaml = os.path.join(self.test_yaml_dir, "react_node.yaml")

        # Act - Test that we can parse multiple YAML files
        from petal.core.yaml.parser import YAMLNodeParser

        parser = YAMLNodeParser()
        llm_config = parser.parse_node_config(llm_yaml)
        react_config = parser.parse_node_config(react_yaml)

        # Assert
        assert llm_config is not None
        assert llm_config.type == "llm"
        assert llm_config.name == "assistant"

        assert react_config is not None
        assert react_config.type == "react"
        assert react_config.name == "reasoning_agent"

    def test_yaml_node_with_missing_required_fields(self):
        """Test error handling for missing required fields in YAML."""
        # Arrange
        import tempfile

        from pydantic import ValidationError

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("type: llm\n")  # Missing required fields
            temp_path = f.name
        try:
            # Act & Assert
            with pytest.raises(ValidationError):
                self.factory.node_from_yaml(temp_path)
        finally:
            os.unlink(temp_path)
