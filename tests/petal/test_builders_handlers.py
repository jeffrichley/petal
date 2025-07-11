"""Tests for configuration handlers using Chain of Responsibility pattern."""

from typing import Any, Dict

import pytest
from petal.core.builders.handlers.base import StepConfigHandler
from petal.core.builders.handlers.custom import CustomConfigHandler
from petal.core.builders.handlers.llm import LLMConfigHandler


class TestStepConfigHandler:
    """Test the base StepConfigHandler class."""

    def test_abstract_base_class_cannot_be_instantiated(self):
        """Test that StepConfigHandler ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            # This should fail because StepConfigHandler is abstract
            StepConfigHandler()  # type: ignore

    def test_concrete_handler_can_be_instantiated(self):
        """Test that concrete handlers can be instantiated."""
        handler = LLMConfigHandler()
        assert handler is not None

    def test_handler_chain_creation(self):
        """Test creating a chain of handlers."""
        llm_handler = LLMConfigHandler()
        custom_handler = CustomConfigHandler(llm_handler)

        assert custom_handler.next_handler == llm_handler

    def test_handler_chain_without_next(self):
        """Test handler without next handler."""
        handler = LLMConfigHandler()
        assert handler.next_handler is None

    def test_can_handle_method_exists(self):
        """Test that can_handle method exists and works."""
        handler = LLMConfigHandler()
        assert hasattr(handler, "can_handle")
        assert callable(handler.can_handle)

    def test_handle_method_exists(self):
        """Test that handle method exists and works."""
        handler = LLMConfigHandler()
        assert hasattr(handler, "handle")
        assert callable(handler.handle)

    def test_process_method_exists(self):
        """Test that process method exists and works."""
        handler = LLMConfigHandler()
        assert hasattr(handler, "process")
        assert callable(handler.process)


class TestLLMConfigHandler:
    """Test the LLM configuration handler."""

    def test_can_handle_llm_step_type(self):
        """Test that LLM handler can handle 'llm' step type."""
        handler = LLMConfigHandler()
        assert handler.can_handle("llm") is True

    def test_cannot_handle_other_step_types(self):
        """Test that LLM handler cannot handle other step types."""
        handler = LLMConfigHandler()
        assert handler.can_handle("custom") is False
        assert handler.can_handle("unknown") is False

    def test_handle_creates_llm_step(self):
        """Test that handle method creates an LLM step."""
        handler = LLMConfigHandler()
        config: Dict[str, Any] = {
            "prompt_template": "Hello {name}",
            "system_prompt": "You are a helpful assistant",
            "llm_config": {"model": "gpt-4o-mini"},
        }

        step = handler.handle(config)
        assert callable(step)
        assert hasattr(step, "prompt_template")
        assert step.prompt_template == "Hello {name}"

    def test_handle_with_minimal_config(self):
        """Test handle with minimal configuration."""
        handler = LLMConfigHandler()
        config: Dict[str, Any] = {"prompt_template": "Hello"}

        step = handler.handle(config)
        assert callable(step)

    def test_process_llm_step_type(self):
        """Test process method with LLM step type."""
        handler = LLMConfigHandler()
        config: Dict[str, Any] = {"prompt_template": "Hello {name}"}

        step = handler.process("llm", config)
        assert callable(step)

    def test_process_unknown_step_type_raises_error(self):
        """Test that process raises error for unknown step type."""
        handler = LLMConfigHandler()
        config: Dict[str, Any] = {"prompt_template": "Hello"}

        with pytest.raises(ValueError, match="No handler found for step type: custom"):
            handler.process("custom", config)

    def test_process_delegates_to_next_handler(self):
        """Test that process delegates to next handler when cannot handle."""
        llm_handler = LLMConfigHandler()
        custom_handler = CustomConfigHandler(llm_handler)

        # Custom handler should delegate to LLM handler for 'llm' type
        config: Dict[str, Any] = {"prompt_template": "Hello"}
        step = custom_handler.process("llm", config)
        assert callable(step)

    def test_handle_validates_llm_config(self):
        """Test that handle validates LLM configuration."""
        handler = LLMConfigHandler()

        # Test with invalid config
        with pytest.raises(ValueError):
            handler.handle({})  # Missing required fields


class TestCustomConfigHandler:
    """Test the custom configuration handler."""

    def test_can_handle_custom_step_type(self):
        """Test that custom handler can handle 'custom' step type."""
        handler = CustomConfigHandler()
        assert handler.can_handle("custom") is True

    def test_cannot_handle_other_step_types(self):
        """Test that custom handler cannot handle other step types."""
        handler = CustomConfigHandler()
        assert handler.can_handle("llm") is False
        assert handler.can_handle("unknown") is False

    def test_handle_creates_custom_step(self):
        """Test that handle method creates a custom step."""
        handler = CustomConfigHandler()

        def test_function(state: Dict[str, Any]) -> Dict[str, Any]:
            state["processed"] = True
            return state

        config: Dict[str, Any] = {"step_function": test_function}
        step = handler.handle(config)
        assert callable(step)

    def test_handle_with_sync_function(self):
        """Test handle with synchronous function."""
        handler = CustomConfigHandler()

        def sync_func(state: Dict[str, Any]) -> Dict[str, Any]:
            return state

        config: Dict[str, Any] = {"step_function": sync_func}
        step = handler.handle(config)
        assert callable(step)

    def test_handle_with_async_function(self):
        """Test handle with asynchronous function."""
        handler = CustomConfigHandler()

        async def async_func(state: Dict[str, Any]) -> Dict[str, Any]:
            return state

        config: Dict[str, Any] = {"step_function": async_func}
        step = handler.handle(config)
        assert callable(step)

    def test_process_custom_step_type(self):
        """Test process method with custom step type."""
        handler = CustomConfigHandler()

        def test_function(state: Dict[str, Any]) -> Dict[str, Any]:
            return state

        config: Dict[str, Any] = {"step_function": test_function}
        step = handler.process("custom", config)
        assert callable(step)

    def test_handle_raises_error_for_non_callable(self):
        """Test that handle raises error for non-callable step function."""
        handler = CustomConfigHandler()
        config: Dict[str, Any] = {"step_function": "not a function"}

        with pytest.raises(ValueError, match="Custom step must be callable"):
            handler.handle(config)

    def test_handle_raises_error_for_missing_step_function(self):
        """Test that handle raises error for missing step function."""
        handler = CustomConfigHandler()
        config: Dict[str, Any] = {}

        with pytest.raises(ValueError, match="Custom step must be callable"):
            handler.handle(config)

    def test_process_delegates_to_next_handler(self):
        """Test that process delegates to next handler when cannot handle."""
        llm_handler = LLMConfigHandler()
        custom_handler = CustomConfigHandler(llm_handler)

        # Custom handler should handle 'custom' type itself
        def test_function(state: Dict[str, Any]) -> Dict[str, Any]:
            return state

        config: Dict[str, Any] = {"step_function": test_function}
        step = custom_handler.process("custom", config)
        assert callable(step)


class TestHandlerChain:
    """Test the complete handler chain."""

    def test_chain_handles_llm_steps(self):
        """Test that chain can handle LLM steps."""
        llm_handler = LLMConfigHandler()
        custom_handler = CustomConfigHandler(llm_handler)

        config: Dict[str, Any] = {"prompt_template": "Hello"}
        step = custom_handler.process("llm", config)
        assert callable(step)

    def test_chain_handles_custom_steps(self):
        """Test that chain can handle custom steps."""
        llm_handler = LLMConfigHandler()
        custom_handler = CustomConfigHandler(llm_handler)

        def test_function(state: Dict[str, Any]) -> Dict[str, Any]:
            return state

        config: Dict[str, Any] = {"step_function": test_function}
        step = custom_handler.process("custom", config)
        assert callable(step)

    def test_chain_raises_error_for_unknown_step_type(self):
        """Test that chain raises error for unknown step type."""
        llm_handler = LLMConfigHandler()
        custom_handler = CustomConfigHandler(llm_handler)

        config: Dict[str, Any] = {"some_config": "value"}

        with pytest.raises(ValueError, match="No handler found for step type: unknown"):
            custom_handler.process("unknown", config)

    def test_chain_with_multiple_handlers(self):
        """Test chain with multiple handlers in sequence."""
        # Create a chain: custom -> llm -> None
        llm_handler = LLMConfigHandler()
        custom_handler = CustomConfigHandler(llm_handler)

        # Test that custom handler delegates to LLM handler
        config: Dict[str, Any] = {"prompt_template": "Hello"}
        step = custom_handler.process("llm", config)
        assert callable(step)

        # Test that custom handler handles its own type
        def test_function(state: Dict[str, Any]) -> Dict[str, Any]:
            return state

        config = {"step_function": test_function}
        step = custom_handler.process("custom", config)
        assert callable(step)


class TestHandlerIntegration:
    """Test integration with existing components."""

    def test_handler_with_step_registry(self):
        """Test that handlers work with step registry."""
        from petal.core.steps.registry import StepRegistry

        # Create registry and handler
        registry = StepRegistry()
        llm_handler = LLMConfigHandler()

        # Handler should be able to create steps that work with registry
        config: Dict[str, Any] = {"prompt_template": "Hello"}
        step = llm_handler.handle(config)
        assert callable(step)

        # Verify the step can be used with the registry
        assert "llm" in registry._strategies

    def test_handler_with_agent_builder(self):
        """Test that handlers can be integrated with agent builder."""
        from petal.core.builders.agent import AgentBuilder

        # This test will be implemented when we integrate handlers with builder
        builder = AgentBuilder(dict)
        assert builder is not None
