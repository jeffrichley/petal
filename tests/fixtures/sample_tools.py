from langchain_core.tools import tool


@tool
def echo_tool(text: str) -> str:
    """Echoes the input text."""
    return text


def regular_function(text: str) -> str:
    """A regular function without @tool decorator."""
    return f"Regular: {text}"
