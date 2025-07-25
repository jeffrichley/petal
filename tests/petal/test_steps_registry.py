"""Tests for the StepRegistry class."""

from typing import Any, Dict

import pytest

from petal.core.config.agent import StepConfig
from petal.core.steps.base import StepStrategy
from petal.core.steps.custom import CustomStepStrategy
from petal.core.steps.registry import StepRegistry


class DummyStrategy(StepStrategy):
    async def create_step(self, config: Dict[str, Any]):  # noqa: ARG002
        return lambda x: x

    def get_node_name(self, index: int) -> str:
        return f"dummy_{index}"


def test_register_and_retrieve_strategy():
    registry = StepRegistry()
    registry.register("custom", CustomStepStrategy)
    strategy = registry.get_strategy("custom")
    assert isinstance(strategy, CustomStepStrategy)
    # Should be a new instance each time
    strategy2 = registry.get_strategy("custom")
    assert strategy is not strategy2


def test_register_and_retrieve_multiple_strategies():
    registry = StepRegistry()
    registry.register("custom", CustomStepStrategy)
    registry.register("dummy", DummyStrategy)
    assert isinstance(registry.get_strategy("custom"), CustomStepStrategy)
    assert isinstance(registry.get_strategy("dummy"), DummyStrategy)


def test_error_on_unknown_strategy():
    registry = StepRegistry()
    with pytest.raises(ValueError, match="Unknown step type: unknown"):
        registry.get_strategy("unknown")


def test_register_overwrites_existing():
    registry = StepRegistry()
    registry.register("custom", CustomStepStrategy)
    registry.register("custom", DummyStrategy)
    # Should now return DummyStrategy
    assert isinstance(registry.get_strategy("custom"), DummyStrategy)


def test_register_defaults_stub():
    registry = StepRegistry()
    # _register_defaults is called in __init__, should not raise
    # For now, just check that registry is usable
    registry.register("custom", CustomStepStrategy)
    assert isinstance(registry.get_strategy("custom"), CustomStepStrategy)


def test_default_strategy_is_registered():
    """Test that CustomStepStrategy is registered as a default strategy."""
    registry = StepRegistry()
    # CustomStepStrategy should be available as "custom" by default
    strategy = registry.get_strategy("custom")
    assert isinstance(strategy, CustomStepStrategy)


def test_thread_safety():
    import threading

    registry = StepRegistry()

    def register_custom():
        for _ in range(100):
            registry.register("custom", CustomStepStrategy)

    threads = [threading.Thread(target=register_custom) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Should not raise, and should be retrievable
    assert isinstance(registry.get_strategy("custom"), CustomStepStrategy)


def test_validate_strategy_success():
    """Test that validate_strategy succeeds for registered strategies."""
    registry = StepRegistry()
    registry.register("custom", CustomStepStrategy)

    # Should not raise
    registry.validate_strategy("custom")


def test_validate_strategy_failure():
    """Test that validate_strategy raises error for unregistered strategies."""
    registry = StepRegistry()

    with pytest.raises(ValueError, match="Unknown step type: unknown"):
        registry.validate_strategy("unknown")


def test_validate_strategy_with_defaults():
    """Test that validate_strategy works with default strategies."""
    registry = StepRegistry()

    # Default strategies should be valid
    registry.validate_strategy("custom")
    registry.validate_strategy("llm")

    # Unknown strategies should fail
    with pytest.raises(ValueError, match="Unknown step type: unknown"):
        registry.validate_strategy("unknown")


@pytest.mark.asyncio
async def test_create_step_with_valid_config():
    registry = StepRegistry()
    registry.register("dummy", DummyStrategy)

    step_config = StepConfig(
        strategy_type="dummy", config={"test": "value"}, node_name="test_node"
    )

    step = await registry.create_step(step_config)
    assert callable(step)

    # Test that the step can be called
    result = step({"input": "test"})
    assert result == {"input": "test"}


@pytest.mark.asyncio
async def test_create_step_with_unknown_strategy():
    """Test that create_step raises error for unknown strategy type."""
    registry = StepRegistry()

    step_config = StepConfig(
        strategy_type="unknown", config={}, node_name="unknown_node"
    )

    with pytest.raises(ValueError, match="Unknown step type: unknown"):
        await registry.create_step(step_config)


@pytest.mark.asyncio
async def test_create_step_with_custom_strategy():
    """Test that create_step works with CustomStepStrategy."""
    registry = StepRegistry()

    step_config = StepConfig(
        strategy_type="custom",
        config={"step_function": lambda x: {"result": x["input"] * 2}},
        node_name="custom_node",
    )

    step = await registry.create_step(step_config)
    assert callable(step)

    # Test that the step can be called
    result = step({"input": 5})
    assert result == {"result": 10}
