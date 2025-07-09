"""Tests for the StepStrategy base class."""

from abc import ABC
from typing import Any, Callable, Dict

import pytest
from petal.core.steps.base import MyCustomStrategy, StepStrategy


class TestStepStrategyABC:
    """Test the StepStrategy abstract base class."""

    def test_abc_cannot_be_instantiated(self):
        """Test that StepStrategy ABC cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            StepStrategy()  # type: ignore

    def test_abc_has_required_abstract_methods(self):
        """Test that StepStrategy has the required abstract methods."""
        # Check that abstract methods exist
        assert hasattr(StepStrategy, "create_step")
        assert hasattr(StepStrategy, "get_node_name")

        # Check that they are abstract
        assert StepStrategy.create_step.__isabstractmethod__  # type: ignore[attr-defined]
        assert StepStrategy.get_node_name.__isabstractmethod__  # type: ignore[attr-defined]

    def test_concrete_implementation_works(self):
        """Test that a concrete implementation of StepStrategy works correctly."""
        strategy = MyCustomStrategy()

        # Test create_step method
        test_config = {"step_function": lambda x: x}
        step = strategy.create_step(test_config)
        assert callable(step)

        # Test get_node_name method
        node_name = strategy.get_node_name(0)
        assert node_name == "custom_step_0"

        node_name = strategy.get_node_name(5)
        assert node_name == "custom_step_5"

    def test_concrete_implementation_with_different_configs(self):
        """Test concrete implementation with various configuration types."""
        strategy = MyCustomStrategy()

        # Test with different function types
        def test_func(state):  # noqa: ARG001
            return {"processed": True}

        config = {"step_function": test_func}
        step = strategy.create_step(config)
        assert callable(step)

        # Test that the step can be called
        result = step({"input": "test"})
        assert result == {"processed": True}

    def test_node_name_generation_with_different_indices(self):
        """Test node name generation with various indices."""
        strategy = MyCustomStrategy()

        assert strategy.get_node_name(0) == "custom_step_0"
        assert strategy.get_node_name(1) == "custom_step_1"
        assert strategy.get_node_name(10) == "custom_step_10"
        assert strategy.get_node_name(-1) == "custom_step_-1"

    def test_create_step_with_invalid_config(self):
        """Test create_step with invalid configuration."""
        strategy = MyCustomStrategy()

        # Test with missing required key
        with pytest.raises(KeyError):
            strategy.create_step({})

        # Test with non-callable function
        with pytest.raises(ValueError, match="step_function must be callable"):
            strategy.create_step({"step_function": "not_callable"})

    def test_create_step_with_none_function(self):
        """Test create_step with None as function."""
        strategy = MyCustomStrategy()

        with pytest.raises(ValueError, match="step_function must be callable"):
            strategy.create_step({"step_function": None})

    def test_create_step_with_async_function(self):
        """Test create_step with async function."""
        strategy = MyCustomStrategy()

        async def async_func(state):  # noqa: ARG001
            return {"async_processed": True}

        config = {"step_function": async_func}
        step = strategy.create_step(config)
        assert callable(step)


class TestStepStrategyTypeHints:
    """Test that StepStrategy has proper type hints."""

    def test_create_step_type_hints(self):
        """Test that create_step has proper type hints."""
        import inspect

        sig = inspect.signature(StepStrategy.create_step)
        assert sig.parameters["config"].annotation == Dict[str, Any]
        assert sig.return_annotation == Callable

    def test_get_node_name_type_hints(self):
        """Test that get_node_name has proper type hints."""
        import inspect

        sig = inspect.signature(StepStrategy.get_node_name)
        assert sig.parameters["index"].annotation is int
        assert sig.return_annotation is str


class TestStepStrategyDocstrings:
    """Test that StepStrategy has proper docstrings."""

    def test_class_docstring(self):
        """Test that StepStrategy has a class docstring."""
        assert StepStrategy.__doc__ is not None
        assert (
            "Abstract base class for step creation strategies" in StepStrategy.__doc__
        )

    def test_method_docstrings(self):
        """Test that StepStrategy methods have docstrings."""
        assert StepStrategy.create_step.__doc__ is not None
        assert StepStrategy.get_node_name.__doc__ is not None

        # Check that docstrings contain expected content
        assert (
            "Create a step callable from configuration"
            in StepStrategy.create_step.__doc__
        )
        assert "Generate a node name for the step" in StepStrategy.get_node_name.__doc__


class TestMyCustomStrategy:
    """Test the concrete MyCustomStrategy implementation."""

    def test_my_custom_strategy_instantiation(self):
        """Test that MyCustomStrategy can be instantiated."""
        strategy = MyCustomStrategy()
        assert isinstance(strategy, StepStrategy)
        assert isinstance(strategy, MyCustomStrategy)

    def test_my_custom_strategy_inheritance(self):
        """Test that MyCustomStrategy properly inherits from StepStrategy."""
        assert issubclass(MyCustomStrategy, StepStrategy)
        assert issubclass(MyCustomStrategy, ABC)

    def test_my_custom_strategy_methods(self):
        """Test that MyCustomStrategy implements all required methods."""
        strategy = MyCustomStrategy()

        # Test that methods are not abstract
        assert not hasattr(strategy.create_step, "__isabstractmethod__")
        assert not hasattr(strategy.get_node_name, "__isabstractmethod__")

    def test_my_custom_strategy_functionality(self):
        """Test the complete functionality of MyCustomStrategy."""
        strategy = MyCustomStrategy()

        def test_function(state):
            return {"result": state.get("input", "default")}

        # Test create_step
        config = {"step_function": test_function}
        step = strategy.create_step(config)

        # Test that step works
        result = step({"input": "test_value"})
        assert result == {"result": "test_value"}

        # Test get_node_name
        node_name = strategy.get_node_name(3)
        assert node_name == "custom_step_3"
