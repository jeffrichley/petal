import pytest
from petal.core.factory import AgentFactory

from tests.petal.conftest_factory import ChatState


# New tests for ToolRegistry integration
class TestToolRegistryIntegration:
    """Test AgentFactory integration with ToolRegistry singleton."""

    def test_agent_factory_uses_tool_registry_singleton(self):
        """AgentFactory should use ToolRegistry singleton for tool resolution."""
        # Arrange
        from petal.core.registry import ToolRegistry

        factory1 = AgentFactory(ChatState)
        factory2 = AgentFactory(ChatState)

        # Act & Assert
        assert (
            factory1._tool_registry is factory2._tool_registry
        )  # Same singleton instance
        assert isinstance(factory1._tool_registry, ToolRegistry)
        assert not hasattr(factory1, "_tool_factory")  # No longer uses ToolFactory

    def test_agent_factory_tool_discovery_configuration(self):
        """AgentFactory should support tool discovery configuration."""
        # Arrange
        factory = AgentFactory(ChatState)

        # Act
        factory.with_tool_discovery(
            enabled=True, folders=["custom_tools/"], exclude_patterns=["test_*"]
        )

        # Assert
        assert factory._tool_discovery_config["enabled"] is True
        assert factory._tool_discovery_config["folders"] == ["custom_tools/"]
        assert factory._tool_discovery_config["exclude_patterns"] == ["test_*"]

    @pytest.mark.asyncio
    async def test_agent_factory_backward_compatibility(self):
        """AgentFactory should maintain backward compatibility with existing API."""
        # Arrange
        from langchain_core.tools import BaseTool
        from petal.core.registry import ToolRegistry

        class DummyTool(BaseTool):
            name: str = "test_tool"
            description: str = "A dummy tool for testing."

            def _run(self, *args, **kwargs):  # noqa: ARG002
                return "ok"

            async def _arun(self, *args, **kwargs):  # noqa: ARG002
                return "ok"

        ToolRegistry().add("test_tool", DummyTool())

        factory = AgentFactory(ChatState)

        # Act - existing API should work unchanged
        agent = await (
            factory.with_chat(provider="openai", model="gpt-4o-mini")
            .with_tools(["test_tool"])
            .build()
        )

        # Assert
        assert agent is not None
        assert hasattr(agent, "arun")
        # No exceptions should be raised

    @pytest.mark.asyncio
    async def test_agent_factory_tool_discovery_chain(self):
        """AgentFactory should use ToolRegistry's discovery chain for missing tools."""
        # Arrange
        factory = AgentFactory(ChatState)
        factory.with_chat(provider="openai", model="gpt-4o-mini")

        # Act - tool not in registry, should trigger discovery
        try:
            factory.with_tools(["nonexistent_tool"])
            # Should either find tool via discovery or raise appropriate error
        except KeyError as e:
            # Expected if tool not found after discovery attempts
            assert "nonexistent_tool" in str(e)

    def test_agent_factory_tool_discovery_default_config(self):
        """AgentFactory should have sensible defaults for tool discovery."""
        # Arrange
        factory = AgentFactory(ChatState)

        # Act - should have default config even without explicit configuration
        assert hasattr(factory, "_tool_discovery_config")
        assert factory._tool_discovery_config["enabled"] is True  # Default enabled

    def test_agent_factory_tool_discovery_fluent_interface(self):
        """AgentFactory tool discovery should support fluent interface."""
        # Arrange
        factory = AgentFactory(ChatState)

        # Act
        result = factory.with_tool_discovery(enabled=False, folders=["my_tools/"])

        # Assert
        assert result is factory  # Should return self for chaining
        assert factory._tool_discovery_config["enabled"] is False
        assert factory._tool_discovery_config["folders"] == ["my_tools/"]
