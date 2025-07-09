"""State type factory for creating state types with message support."""

import types
from typing import Dict, Type, TypedDict, get_type_hints

from langgraph.graph.message import add_messages
from typing_extensions import Annotated


class StateTypeFactory:
    """Factory for creating state types with message support."""

    _cache: Dict[tuple, Type] = {}

    @classmethod
    def create_with_messages(cls, base_type: Type) -> Type:
        """
        Create a state type that includes message support.

        Args:
            base_type: The base state type to enhance with message support

        Returns:
            Type: The enhanced state type with message support

        Raises:
            ValueError: If the base_type is None or not a TypedDict
        """
        if base_type is None:
            raise ValueError("Invalid state type: None")

        # Check if it's a TypedDict (or compatible type)
        if not hasattr(base_type, "__annotations__"):
            raise ValueError("State type must be a TypedDict or compatible type")

        # Additional check for TypedDict types
        if not hasattr(base_type, "__total__"):
            raise ValueError("State type must be a TypedDict")

        # Get type hints to check if messages field already exists
        type_hints = get_type_hints(base_type, include_extras=True)

        # If messages field already exists, return the original type
        if "messages" in type_hints:
            return base_type

        # Create cache key based on type name and type hints
        cache_key = (base_type.__name__, tuple(sorted(type_hints.items())))

        # Check cache first
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        # Create new TypedDict with all original fields plus messages
        dynamic_name = f"{base_type.__name__}WithMessagesAddedByPetal"

        # Create a new TypedDict class with all the original annotations plus messages
        new_annotations = dict(type_hints)
        new_annotations["messages"] = Annotated[list, add_messages]

        # Create the new type using type() and TypedDict as base
        combined_type = types.new_class(
            dynamic_name,
            (TypedDict,),
            exec_body=lambda ns: ns.update({"__annotations__": new_annotations}),
        )

        # Cache the result
        cls._cache[cache_key] = combined_type

        return combined_type

    @classmethod
    def create_mergeable(cls, base_type: Type) -> Type:
        """
        Create a state type that supports merging.

        Args:
            base_type: The base state type to make mergeable

        Returns:
            Type: The mergeable state type (currently returns original type)
        """
        # For now, return the original type
        # This method is for future extensibility when mergeable state types
        # are implemented
        return base_type

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the internal cache. Useful for testing."""
        cls._cache.clear()
