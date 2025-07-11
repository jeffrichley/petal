"""Tests for StateTypeFactory."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from typing import TypedDict, get_type_hints

import pytest
from langgraph.graph.message import add_messages
from petal.core.config.state import StateTypeFactory
from typing_extensions import Annotated


class SimpleState(TypedDict):
    """Simple state for testing."""

    name: str
    value: int


class StateWithMessages(TypedDict):
    """State that already has messages field."""

    messages: Annotated[list, add_messages]
    name: str


class ComplexState(TypedDict):
    """Complex state with multiple fields."""

    name: str
    age: int
    preferences: dict
    metadata: list


def test_state_type_factory_create_with_messages_basic():
    """Test basic state type creation with messages."""
    # Test with simple state that doesn't have messages
    result_type = StateTypeFactory.create_with_messages(SimpleState)

    # Should be a different type (not the same object)
    assert result_type is not SimpleState

    # Should have messages field from MessagesState
    type_hints = get_type_hints(result_type, include_extras=True)
    assert "messages" in type_hints

    # Should preserve original fields
    assert "name" in type_hints
    assert "value" in type_hints


def test_state_type_factory_create_with_messages_already_has_messages():
    """Test state type that already has messages field."""
    # Test with state that already has messages
    result_type = StateTypeFactory.create_with_messages(StateWithMessages)

    # Should return the original type unchanged
    assert result_type is StateWithMessages

    # Should still have messages field
    type_hints = get_type_hints(result_type, include_extras=True)
    assert "messages" in type_hints


def test_state_type_factory_create_with_messages_caching():
    """Test that caching works correctly."""
    # Create the same type twice
    type1 = StateTypeFactory.create_with_messages(SimpleState)
    type2 = StateTypeFactory.create_with_messages(SimpleState)

    # Should return the same cached type
    assert type1 is type2

    # Should have the same name pattern
    assert "SimpleStateWithMessagesAddedByPetal" in type1.__name__


def test_state_type_factory_create_with_messages_complex_state():
    """Test with complex state type."""
    result_type = StateTypeFactory.create_with_messages(ComplexState)

    # Should be a different type (not the same object)
    assert result_type is not ComplexState

    # Should preserve all original fields
    type_hints = get_type_hints(result_type, include_extras=True)
    assert "name" in type_hints
    assert "age" in type_hints
    assert "preferences" in type_hints
    assert "metadata" in type_hints
    assert "messages" in type_hints


def test_state_type_factory_create_with_messages_different_instances():
    """Test that different state types create different cached types."""
    type1 = StateTypeFactory.create_with_messages(SimpleState)
    type2 = StateTypeFactory.create_with_messages(ComplexState)

    # Should be different types
    assert type1 is not type2

    # Should have different names
    assert type1.__name__ != type2.__name__


def test_state_type_factory_create_mergeable_basic():
    """Test basic mergeable state type creation."""
    result_type = StateTypeFactory.create_mergeable(SimpleState)

    # For now, should return the original type
    # This method is for future extensibility
    assert result_type is SimpleState


def test_state_type_factory_create_mergeable_complex():
    """Test mergeable state type with complex state."""
    result_type = StateTypeFactory.create_mergeable(ComplexState)

    # Should return the original type
    assert result_type is ComplexState


def test_state_type_factory_error_handling_invalid_type():
    """Test error handling for invalid state types."""
    with pytest.raises(ValueError, match="Invalid state type"):
        StateTypeFactory.create_with_messages(None)  # type: ignore[arg-type]


def test_state_type_factory_error_handling_not_typed_dict():
    """Test error handling for non-TypedDict types."""

    class RegularClass:
        pass

    with pytest.raises(ValueError, match="State type must be a TypedDict"):
        StateTypeFactory.create_with_messages(RegularClass)


def test_state_type_factory_type_hints_preserved():
    """Test that type hints are preserved in created types."""
    result_type = StateTypeFactory.create_with_messages(SimpleState)

    type_hints = get_type_hints(result_type, include_extras=True)

    # Should preserve original type hints
    assert type_hints["name"] is str
    assert type_hints["value"] is int
    assert "messages" in type_hints


def test_state_type_factory_dynamic_name_generation():
    """Test that dynamic names are generated correctly."""
    result_type = StateTypeFactory.create_with_messages(SimpleState)

    # Should follow the naming pattern
    assert result_type.__name__.startswith("SimpleState")
    assert result_type.__name__.endswith("WithMessagesAddedByPetal")


def test_state_type_factory_cache_key_uniqueness():
    """Test that cache keys are unique for different state types."""

    # Create different state types
    class StateA(TypedDict):
        field1: str

    class StateB(TypedDict):
        field2: int

    type_a = StateTypeFactory.create_with_messages(StateA)
    type_b = StateTypeFactory.create_with_messages(StateB)

    # Should be different types
    assert type_a is not type_b
    assert type_a.__name__ != type_b.__name__


def test_state_type_factory_cache_key_same_type():
    """Test that cache keys are the same for identical state types."""
    # Create the same type multiple times
    type1 = StateTypeFactory.create_with_messages(SimpleState)
    type2 = StateTypeFactory.create_with_messages(SimpleState)
    type3 = StateTypeFactory.create_with_messages(SimpleState)

    # All should be the same cached type
    assert type1 is type2
    assert type2 is type3
    assert type1 is type3


def test_state_type_factory_clear_cache():
    """Test that clear_cache method works correctly."""
    # Create a type to populate cache
    StateTypeFactory.create_with_messages(SimpleState)

    # Verify cache has entry
    assert len(StateTypeFactory._cache) > 0

    # Clear cache
    StateTypeFactory.clear_cache()

    # Verify cache is empty
    assert len(StateTypeFactory._cache) == 0


def test_state_type_factory_typed_dict_validation():
    """Test that __total__ check works for TypedDict validation."""

    # Create a class that has __annotations__ but not __total__ (not a TypedDict)
    class FakeTypedDict:
        __annotations__ = {"field": str}
        # Missing __total__ attribute

    with pytest.raises(ValueError, match="State type must be a TypedDict"):
        StateTypeFactory.create_with_messages(FakeTypedDict)


def test_state_type_factory_typed_dict_validation_missing_total():
    """Test that __total__ check works for TypedDict validation."""

    # Create a class that has __annotations__ but not __total__ (not a TypedDict)
    class FakeTypedDict:
        __annotations__ = {"field": str}
        # Missing __total__ attribute

    with pytest.raises(ValueError, match="State type must be a TypedDict"):
        StateTypeFactory.create_with_messages(FakeTypedDict)


def test_state_type_factory_error_handling_missing_annotations():
    """Test error handling for types missing __annotations__."""

    class NoAnnotations:
        pass

    with pytest.raises(ValueError, match="State type must be a TypedDict"):
        StateTypeFactory.create_with_messages(NoAnnotations)


def test_state_type_factory_error_handling_no_annotations_attribute():
    """Test error handling for objects with no __annotations__ attribute."""
    with pytest.raises(
        ValueError, match="State type must be a TypedDict or compatible type"
    ):
        StateTypeFactory.create_with_messages(42)  # type: ignore[arg-type]  # int has no __annotations__
