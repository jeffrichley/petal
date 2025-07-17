"""Shared fixtures and state types for AgentFactory tests."""

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# Common state types used across multiple test files
class SimpleState(TypedDict):
    x: int
    processed: bool


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


class MixedState(TypedDict):
    messages: Annotated[list, add_messages]
    processed: bool
    x: int


class DefaultState(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    answer: str
