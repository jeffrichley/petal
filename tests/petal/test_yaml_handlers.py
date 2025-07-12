from unittest.mock import patch

import pytest
from petal.core.config.yaml import BaseNodeConfig, LLMNodeConfig, ReactNodeConfig
from petal.core.yaml.handlers import HandlerFactory
from petal.core.yaml.handlers.base import NodeConfigHandler
from petal.core.yaml.handlers.llm import LLMNodeHandler
from petal.core.yaml.handlers.react import ReactNodeHandler


def test_node_config_handler_abc():
    """Test that NodeConfigHandler is an abstract base class."""

    class IncompleteHandler(NodeConfigHandler):
        pass

    with pytest.raises(TypeError):
        # Should fail - subclass without create_node implementation
        IncompleteHandler()  # type: ignore[abstract]


def test_node_config_handler_interface():
    """Test that NodeConfigHandler defines required interface."""

    class TestHandler(NodeConfigHandler):
        def create_node(self, config: BaseNodeConfig):
            # Use config to avoid unused argument warning
            _ = config
            return lambda x: x

    handler = TestHandler()
    assert hasattr(handler, "create_node")

    # Test that create_node method works
    config = LLMNodeConfig(
        type="llm",
        name="test",
        description="Test LLM node",
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
        prompt="Test prompt",
        system_prompt="Test system",
    )
    node = handler.create_node(config)
    assert callable(node)


def test_llm_node_handler_creation():
    """Test LLMNodeHandler can create LLM nodes from config."""
    handler = LLMNodeHandler()
    config = LLMNodeConfig(
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
    node = handler.create_node(config)
    assert callable(node)
    assert callable(node)


def test_llm_node_handler_integration_with_registry(monkeypatch):
    """Test LLMNodeHandler integrates with step registry."""
    handler = LLMNodeHandler()
    config = LLMNodeConfig(
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
    called = {}

    class DummyRegistry:
        def create_step(self, step_config):
            called["step_config"] = step_config
            return lambda x: x

    monkeypatch.setattr(handler, "registry", DummyRegistry())
    node = handler.create_node(config)
    assert callable(node)
    assert "step_config" in called


def dummy_tool_func(x):
    """A dummy tool for testing."""
    return x


def test_react_node_handler_creation():
    """Test ReactNodeHandler can create React nodes from config."""
    with patch(
        "petal.core.tool_factory.ToolFactory.resolve",
        lambda _self, _name: dummy_tool_func,
    ):
        handler = ReactNodeHandler()
        config = ReactNodeConfig(
            type="react",
            name="test_react",
            description="Test React node",
            tools=["search", "calculator"],
            reasoning_prompt="Think step by step",
            system_prompt="You are a reasoning agent",
            max_iterations=5,
        )
        node = handler.create_node(config)
        assert callable(node)
        assert callable(node)


def test_react_node_handler_with_tool_factory():
    """Test ReactNodeHandler integrates with ToolFactory."""
    with patch(
        "petal.core.tool_factory.ToolFactory.resolve",
        lambda _self, _name: dummy_tool_func,
    ):
        handler = ReactNodeHandler()
        config = ReactNodeConfig(
            type="react",
            name="test_react",
            description="Test React node",
            tools=["search", "calculator"],
            reasoning_prompt="Think step by step",
            system_prompt="You are a reasoning agent",
            max_iterations=5,
        )
        node = handler.create_node(config)
        assert callable(node)
        # No need to check tool resolution here, just that it works


def test_handler_factory_creates_correct_handlers():
    """Test HandlerFactory creates correct handlers for each node type."""
    factory = HandlerFactory()
    llm_handler = factory.get_handler("llm")
    assert isinstance(llm_handler, LLMNodeHandler)
    react_handler = factory.get_handler("react")
    assert isinstance(react_handler, ReactNodeHandler)


def test_handler_factory_unknown_type():
    """Test HandlerFactory raises error for unknown node types."""
    factory = HandlerFactory()
    with pytest.raises(ValueError, match="Unknown node type"):
        factory.get_handler("unknown")


def test_react_node_handler_with_mcp_tools(monkeypatch):
    """Test: ReactNodeHandler should register MCP servers from YAML config using generic resolver."""
    from petal.core.config.yaml import ReactNodeConfig
    from petal.core.tool_factory import ToolFactory
    from petal.core.yaml.handlers.react import ReactNodeHandler

    # Simulate MCP server config in YAML
    mcp_servers = {
        "math_server": {
            "config": {
                "servers": {
                    "math_tools": {
                        "command": "npx -y @modelcontextprotocol/server-math"
                    }
                }
            }
        }
    }
    config = ReactNodeConfig(
        type="react",
        name="test_mcp_react",
        description="Test React node with MCP tools",
        tools=["mcp:math_server:add", "mcp:math_server:multiply"],
        reasoning_prompt="Test reasoning",
        system_prompt="Test system",
        max_iterations=3,
        mcp_servers=mcp_servers,
    )
    # Patch ToolFactory.add_mcp to track calls
    called = {}

    def fake_add_mcp(self, server_name, _resolver=None, mcp_config=None):
        called[server_name] = mcp_config
        return self

    monkeypatch.setattr(ToolFactory, "add_mcp", fake_add_mcp)

    # Patch ToolFactory.resolve to always return a dummy function with a docstring
    def dummy_tool(*_args, **_kwargs):
        """A dummy tool for testing."""
        return None

    monkeypatch.setattr(ToolFactory, "resolve", lambda _self, _name: dummy_tool)
    handler = ReactNodeHandler(tool_factory=ToolFactory())
    # Call handler.create_node, which should trigger MCP registration
    handler.create_node(config)
    assert "math_server" in called
    assert called["math_server"] == mcp_servers["math_server"]["config"]
