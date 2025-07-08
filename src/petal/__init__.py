"""
Petal - Agent and tool creation framework.

A framework for creating AI agents and tools with a clean, modular API.
"""

__version__ = "0.1.0"
__author__ = "Jeff Richley"
__email__ = "jeffrichley@gmail.com"

# Version info
VERSION = __version__


def hello(name: str) -> str:
    """
    Return a friendly greeting.

    Args:
        name (str): The name to greet.

    Returns:
        str: Greeting message.
    """
    if not name:
        raise ValueError("Name must not be empty.")
    return f"Hello, {name}!"
