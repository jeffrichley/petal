import sys
import types
from unittest.mock import patch

import pytest
from petal.core.config.yaml import (
    BaseNodeConfig,
    CustomNodeConfig,
    LLMNodeConfig,
    ReactNodeConfig,
)
from petal.core.yaml.handlers import HandlerFactory
from petal.core.yaml.handlers.base import NodeConfigHandler
from petal.core.yaml.handlers.custom import CustomNodeHandler
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


def test_custom_node_handler_creation():
    """Test CustomNodeHandler can create custom nodes from config."""
    handler = CustomNodeHandler()

    # Create a test function for importing
    def test_function(state, multiplier=1, offset=0):
        """Test function for custom node."""
        return {"result": state.get("value", 0) * multiplier + offset}

    # Mock the import function to return our test function
    with patch.object(handler, "_import_function", return_value=test_function):
        config = CustomNodeConfig(
            type="custom",
            name="test_custom",
            description="Test custom node",
            function_path="test_module.test_function",
            parameters={"multiplier": 2, "offset": 10},
        )

        node = handler.create_node(config)
        assert callable(node)

        # Test the node function
        state = {"value": 5}
        result = node(state)
        assert result == {"result": 20}  # 5 * 2 + 10


def test_custom_node_handler_with_base_config():
    """Test CustomNodeHandler works with BaseNodeConfig input."""
    handler = CustomNodeHandler()

    def test_function(state, **kwargs):
        """Test function for custom node."""
        return {"result": state.get("value", 0) + kwargs.get("add", 0)}

    with patch.object(handler, "_import_function", return_value=test_function):
        # Create a BaseNodeConfig instead of CustomNodeConfig
        base_config = BaseNodeConfig(
            type="custom", name="test_custom", description="Test custom node"
        )
        # Add custom attributes to __dict__ to simulate the conversion
        base_config.__dict__["function_path"] = "test_module.test_function"
        base_config.__dict__["parameters"] = {"add": 5}

        node = handler.create_node(base_config)
        assert callable(node)

        # Test the node function
        state = {"value": 10}
        result = node(state)
        assert result == {"result": 15}  # 10 + 5


def test_custom_node_handler_parameter_merging():
    """Test that custom node parameters are merged with kwargs."""
    handler = CustomNodeHandler()

    def test_function(state, base=0, multiplier=1, **kwargs):
        """Test function that uses parameters."""
        return {"result": (state.get("value", 0) + base) * multiplier, "kwargs": kwargs}

    with patch.object(handler, "_import_function", return_value=test_function):
        config = CustomNodeConfig(
            type="custom",
            name="test_custom",
            description="Test custom node",
            function_path="test_module.test_function",
            parameters={"base": 10, "multiplier": 2},
        )

        node = handler.create_node(config)

        # Test with additional kwargs
        state = {"value": 5}
        result = node(state, extra_param="test", another_param=42)

        assert result["result"] == 30  # (5 + 10) * 2
        assert result["kwargs"]["extra_param"] == "test"
        assert result["kwargs"]["another_param"] == 42


def test_custom_node_handler_empty_parameters():
    """Test custom node handler with empty parameters."""
    handler = CustomNodeHandler()

    def test_function(state, **kwargs):
        """Test function with no default parameters."""
        return {"result": state.get("value", 0), "kwargs": kwargs}

    with patch.object(handler, "_import_function", return_value=test_function):
        config = CustomNodeConfig(
            type="custom",
            name="test_custom",
            description="Test custom node",
            function_path="test_module.test_function",
            parameters={},  # Empty parameters
        )

        node = handler.create_node(config)

        # Test the node function
        state = {"value": 42}
        result = node(state, extra_kwarg="test")

        assert result["result"] == 42
        assert result["kwargs"]["extra_kwarg"] == "test"


def test_custom_node_handler_kwargs_override_parameters():
    """Test that kwargs override config parameters."""
    handler = CustomNodeHandler()

    def test_function(_state, param1="default1", param2="default2"):
        """Test function with default parameters."""
        return {"param1": param1, "param2": param2}

    with patch.object(handler, "_import_function", return_value=test_function):
        config = CustomNodeConfig(
            type="custom",
            name="test_custom",
            description="Test custom node",
            function_path="test_module.test_function",
            parameters={"param1": "config_value", "param2": "config_value2"},
        )

        node = handler.create_node(config)

        # Test that kwargs override config parameters
        state = {"value": 42}
        result = node(state, param1="override_value")

        assert result["param1"] == "override_value"  # Overridden by kwargs
        assert result["param2"] == "config_value2"  # From config


def test_custom_node_handler_import_error():
    """Test CustomNodeHandler handles import errors gracefully."""
    handler = CustomNodeHandler()

    # Mock _import_function to raise ImportError
    with patch.object(
        handler, "_import_function", side_effect=ImportError("Module not found")
    ):
        config = CustomNodeConfig(
            type="custom",
            name="test_custom",
            description="Test custom node",
            function_path="nonexistent.module.function",
            parameters={},
        )

        with pytest.raises(ImportError, match="Module not found"):
            handler.create_node(config)


def test_custom_node_handler_invalid_function_path():
    """Test CustomNodeHandler handles invalid function paths."""
    handler = CustomNodeHandler()

    # Mock _import_function to raise ValueError
    with patch.object(
        handler, "_import_function", side_effect=ValueError("Invalid path")
    ):
        config = CustomNodeConfig(
            type="custom",
            name="test_custom",
            description="Test custom node",
            function_path="invalid.path",
            parameters={},
        )

        with pytest.raises(ValueError, match="Invalid path"):
            handler.create_node(config)


@pytest.mark.asyncio
async def test_custom_node_handler_async_function():
    """Test CustomNodeHandler works with async functions."""
    handler = CustomNodeHandler()

    async def async_test_function(_state, **kwargs):
        """Async test function for custom node."""
        return {"result": kwargs.get("add", 0)}

    with patch.object(handler, "_import_function", return_value=async_test_function):
        config = CustomNodeConfig(
            type="custom",
            name="test_custom",
            description="Test custom node",
            function_path="test_module.async_test_function",
            parameters={"add": 5},
        )

        node = handler.create_node(config)
        assert callable(node)

        # Test the node function
        result = await node({"value": 10})
        assert result["result"] == 5


def test_custom_node_handler_complex_state():
    """Test CustomNodeHandler with complex state objects."""
    handler = CustomNodeHandler()

    def test_function(state, **kwargs):
        """Test function that processes complex state."""
        return {
            "processed": state.get("data", {}).get("items", []),
            "count": len(state.get("data", {}).get("items", [])),
            "kwargs": kwargs,
        }

    with patch.object(handler, "_import_function", return_value=test_function):
        config = CustomNodeConfig(
            type="custom",
            name="test_custom",
            description="Test custom node",
            function_path="test_module.test_function",
            parameters={"filter_type": "active"},
        )

        node = handler.create_node(config)

        # Test with complex state
        state = {
            "data": {"items": [1, 2, 3, 4, 5], "metadata": {"total": 5}},
            "user": {"id": 123},
        }

        result = node(state, filter_type="all")

        assert result["processed"] == [1, 2, 3, 4, 5]
        assert result["count"] == 5
        assert result["kwargs"]["filter_type"] == "all"  # Overridden by kwargs


def test_custom_node_handler_integration_with_factory():
    """Test CustomNodeHandler integrates with HandlerFactory."""
    factory = HandlerFactory()

    # Test that factory can create custom handler
    custom_handler = factory.get_handler("custom")
    assert isinstance(custom_handler, CustomNodeHandler)

    # Test that custom handler is registered
    assert "custom" in factory._handlers


def test_custom_node_handler_edge_cases():
    """Test CustomNodeHandler with edge cases."""
    handler = CustomNodeHandler()

    def test_function(state, **kwargs):
        """Test function for edge cases."""
        return {
            "state_type": type(state).__name__,
            "state_keys": list(state.keys()) if isinstance(state, dict) else [],
            "kwargs_count": len(kwargs),
            "kwargs_keys": list(kwargs.keys()),
        }

    with patch.object(handler, "_import_function", return_value=test_function):
        config = CustomNodeConfig(
            type="custom",
            name="test_custom",
            description="Test custom node",
            function_path="test_module.test_function",
            parameters={"default_param": "default_value"},
        )

        node = handler.create_node(config)

        # Test with None state
        result = node(None)
        assert result["state_type"] == "NoneType"

        # Test with empty dict state
        result = node({})
        assert result["state_type"] == "dict"
        assert result["state_keys"] == []

        # Test with no kwargs
        result = node({"test": "value"})
        assert result["kwargs_count"] == 1  # Only default_param
        assert "default_param" in result["kwargs_keys"]


def test_import_function_valid(monkeypatch):  # noqa: ARG001
    """Test _import_function successfully imports a function from a module path."""
    handler = CustomNodeHandler()
    # Dynamically create a dummy module
    dummy_module = types.ModuleType("dummy_module")

    def dummy_func():
        return "ok"

    dummy_module.dummy_func = dummy_func  # type: ignore[attr-defined]
    sys.modules["dummy_module"] = dummy_module
    try:
        func = handler._import_function("dummy_module.dummy_func")  # type: ignore
        assert callable(func)
        assert func() == "ok"
    finally:
        del sys.modules["dummy_module"]


def test_import_function_invalid_path():
    """Test _import_function raises ValueError for invalid path."""
    handler = CustomNodeHandler()
    with pytest.raises(ValueError, match="Invalid function path"):
        handler._import_function("not_a_path")


def test_import_function_missing_function(monkeypatch):  # noqa: ARG001
    """Test _import_function raises ImportError if function does not exist."""
    handler = CustomNodeHandler()
    dummy_module = types.ModuleType("dummy2")
    sys.modules["dummy2"] = dummy_module
    try:
        with pytest.raises(ImportError, match="not_found.*dummy2"):
            handler._import_function("dummy2.not_found")
    finally:
        del sys.modules["dummy2"]


def test_import_function_not_callable(monkeypatch):  # noqa: ARG001
    """Test _import_function raises ValueError if attribute is not callable."""
    handler = CustomNodeHandler()
    dummy_module = types.ModuleType("dummy3")
    dummy_module.not_callable = 123  # type: ignore[attr-defined]
    sys.modules["dummy3"] = dummy_module
    try:
        with pytest.raises(ValueError, match="not_callable.*callable"):
            handler._import_function("dummy3.not_callable")  # type: ignore
    finally:
        del sys.modules["dummy3"]


def test_custom_node_config_function_path_validation():
    """Test that function_path must be a non-empty string."""
    # Valid path
    config = CustomNodeConfig(
        type="custom",
        name="test",
        description="desc",
        function_path="module.func",
        parameters={},
    )
    assert config.function_path == "module.func"
    # Empty path
    import pytest

    with pytest.raises(ValueError, match="function_path cannot be empty"):
        CustomNodeConfig(
            type="custom",
            name="test",
            description="desc",
            function_path=" ",
            parameters={},
        )


def test_custom_node_config_parameters_validation():
    """Test that parameters must be a dictionary (enforced by Pydantic type system)."""
    # Valid dict
    config = CustomNodeConfig(
        type="custom",
        name="test",
        description="desc",
        function_path="module.func",
        parameters={"a": 1},
    )
    assert config.parameters == {"a": 1}
    # Invalid type - Pydantic will enforce this
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
        CustomNodeConfig(
            type="custom",
            name="test",
            description="desc",
            function_path="module.func",
            parameters=[1, 2, 3],
        )
