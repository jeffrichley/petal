"""Test functions for custom node handler testing."""

import asyncio
from typing import Any, Dict


def simple_function(state: Any, **kwargs) -> Dict[str, Any]:
    """Simple test function that returns state and kwargs."""
    return {"state": state, "kwargs": kwargs}


def math_function(
    state: Dict[str, Any], multiplier: int = 1, offset: int = 0
) -> Dict[str, Any]:
    """Math function that processes numeric state values."""
    value = state.get("value", 0)
    return {"result": value * multiplier + offset}


def complex_state_function(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Function that processes complex state objects."""
    return {
        "processed": state.get("data", {}).get("items", []),
        "count": len(state.get("data", {}).get("items", [])),
        "kwargs": kwargs,
    }


def parameter_test_function(
    _state: Any, param1: str = "default1", param2: str = "default2"
) -> Dict[str, str]:
    """Function to test parameter handling."""
    return {"param1": param1, "param2": param2}


def edge_case_function(state: Any, **kwargs) -> Dict[str, Any]:
    """Function to test edge cases."""
    return {
        "state_type": type(state).__name__,
        "state_keys": list(state.keys()) if isinstance(state, dict) else [],
        "kwargs_count": len(kwargs),
        "kwargs_keys": list(kwargs.keys()),
    }


async def async_function(_state: Any, **kwargs) -> Dict[str, Any]:
    """Async test function."""
    await asyncio.sleep(0.001)  # Simulate async work
    return {"result": kwargs.get("add", 0)}


# Non-callable attribute for testing
NOT_CALLABLE = 123
