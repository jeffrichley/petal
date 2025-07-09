"""Tests for the CustomStepStrategy class."""

from typing import Any, Dict

import pytest
from petal.core.steps.custom import CustomStepStrategy


def dummy_sync(state: Dict[str, Any]) -> Dict[str, Any]:
    state["sync"] = True
    return state


async def dummy_async(state: Dict[str, Any]) -> Dict[str, Any]:
    state["async"] = True
    return state


class CallableClass:
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["called"] = True
        return state


def test_create_step_with_sync_function():
    strategy = CustomStepStrategy()
    config = {"step_function": dummy_sync}
    step = strategy.create_step(config)
    assert callable(step)
    result = step({"a": 1})
    assert result["sync"] is True


def test_create_step_with_async_function():
    strategy = CustomStepStrategy()
    config = {"step_function": dummy_async}
    step = strategy.create_step(config)
    assert callable(step)

    # Can't run async here, just check it's callable


def test_create_step_with_callable_class():
    strategy = CustomStepStrategy()
    config = {"step_function": CallableClass()}
    step = strategy.create_step(config)
    assert callable(step)
    result = step({"b": 2})
    assert result["called"] is True


def test_create_step_with_lambda():
    strategy = CustomStepStrategy()
    config = {"step_function": lambda s: {"ok": bool(s)}}
    step = strategy.create_step(config)
    assert callable(step)
    assert step({"foo": 1}) == {"ok": True}


def test_get_node_name():
    strategy = CustomStepStrategy()
    assert strategy.get_node_name(0) == "custom_step_0"
    assert strategy.get_node_name(5) == "custom_step_5"
    assert strategy.get_node_name(-1) == "custom_step_-1"


def test_create_step_raises_for_non_callable():
    strategy = CustomStepStrategy()
    with pytest.raises(ValueError, match="Custom step must be callable"):
        strategy.create_step({"step_function": "not a function"})
    with pytest.raises(ValueError, match="Custom step must be callable"):
        strategy.create_step({"step_function": None})
    with pytest.raises(ValueError, match="Custom step must be callable"):
        strategy.create_step({})


def test_create_step_with_partial():
    from functools import partial

    strategy = CustomStepStrategy()

    def base(state, x):
        state["x"] = x
        return state

    p = partial(base, x=42)
    config = {"step_function": p}
    step = strategy.create_step(config)
    assert callable(step)
    assert step({})["x"] == 42
