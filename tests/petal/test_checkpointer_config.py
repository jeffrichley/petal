"""Tests for checkpointer configuration."""

import pytest
from pydantic import ValidationError

from petal.core.config.agent import AgentConfig
from petal.core.config.checkpointer import CheckpointerConfig


class TestCheckpointerConfig:
    """Test checkpointer configuration."""

    def test_checkpointer_config_defaults(self):
        """Test checkpointer config with default values."""
        config = CheckpointerConfig()

        assert config.type == "memory"
        assert config.config is None
        assert config.enabled is True

    def test_checkpointer_config_custom_values(self):
        """Test checkpointer config with custom values."""
        config = CheckpointerConfig(
            type="postgres",
            config={"connection_string": "postgresql://..."},
            enabled=False,
        )

        assert config.type == "postgres"
        assert config.config == {"connection_string": "postgresql://..."}
        assert config.enabled is False

    def test_checkpointer_config_invalid_type(self):
        """Test checkpointer config with invalid type."""
        with pytest.raises(ValidationError):
            CheckpointerConfig(type="invalid")  # type: ignore[arg-type]

    def test_agent_config_with_checkpointer(self):
        """Test agent config with checkpointer."""
        checkpointer_config = CheckpointerConfig(type="memory")

        agent_config = AgentConfig(
            name="test_agent",
            state_type=dict,
            memory=None,
            llm_config=None,
            checkpointer=checkpointer_config,
        )

        assert agent_config.checkpointer is not None
        assert agent_config.checkpointer.type == "memory"

    def test_agent_config_set_checkpointer(self):
        """Test setting checkpointer on agent config."""
        agent_config = AgentConfig(
            name="test_agent", state_type=dict, memory=None, llm_config=None
        )

        checkpointer_config = CheckpointerConfig(type="sqlite")
        agent_config.set_checkpointer(checkpointer_config)

        assert agent_config.checkpointer is not None
        assert agent_config.checkpointer.type == "sqlite"
