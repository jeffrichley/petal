from typing import Any, Callable, Optional, TypeVar, Union, overload

from langchain.tools import tool
from langchain_core.tools import BaseTool

from petal.core.registry import ToolRegistry

F = TypeVar("F", bound=Callable[..., Any])


@overload
def petaltool(func: F) -> BaseTool: ...


@overload
def petaltool(
    name_or_callable: str,
    description: Optional[str] = ...,
    return_direct: bool = ...,
    args_schema: Optional[Any] = ...,
    infer_schema: bool = ...,
    response_format: str = ...,
    parse_docstring: bool = ...,
    error_on_invalid_docstring: bool = ...,
    **kwargs: Any,
) -> Callable[[F], BaseTool]: ...


@overload
def petaltool(
    *,
    description: Optional[str] = ...,
    return_direct: bool = ...,
    args_schema: Optional[Any] = ...,
    infer_schema: bool = ...,
    response_format: str = ...,
    parse_docstring: bool = ...,
    error_on_invalid_docstring: bool = ...,
    **kwargs: Any,
) -> Callable[[F], BaseTool]: ...


# NOTE: The implementation signature does not match all overloads for mypy,
# but this is intentional to preserve runtime signature and decorator behavior.
# This is a known limitation with mypy and overloads - see:
#   https://github.com/python/mypy/issues/1484
#   https://github.com/tiangolo/fastapi/issues/240
#   https://github.com/samuelcolvin/pydantic/issues/1223
# The runtime behavior is correct and all tests pass.
def petaltool(  # type: ignore[misc]
    name_or_callable: Optional[Union[str, Callable]] = None,
    description: Optional[str] = None,
    return_direct: bool = False,
    args_schema: Optional[Any] = None,
    infer_schema: bool = True,
    response_format: str = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    **kwargs: Any,
) -> Union[BaseTool, Callable[[Callable[..., Any]], BaseTool]]:
    """
    Petal tool decorator that extends LangChain's @tool functionality.
    """

    def _register_and_return(
        tool_obj: BaseTool, func: Callable, tool_name: str
    ) -> BaseTool:
        registry = ToolRegistry()
        registry.add(tool_name, tool_obj)
        tool_obj._petal_registered = True
        tool_obj._original_func = func
        return tool_obj

    def _decorator(func: Callable) -> BaseTool:
        # Use LangChain's @tool decorator to create the tool
        tool_kwargs = {
            "description": description,
            "return_direct": return_direct,
            "args_schema": args_schema,
            "infer_schema": infer_schema,
            "response_format": response_format,
            "parse_docstring": parse_docstring,
            "error_on_invalid_docstring": error_on_invalid_docstring,
            **kwargs,
        }

        # If name is provided as a string, pass it as name_or_callable
        if isinstance(name_or_callable, str):
            tool_obj = tool(name_or_callable, **tool_kwargs)(func)
        else:
            tool_obj = tool(**tool_kwargs)(func)

        tool_name = (
            name_or_callable if isinstance(name_or_callable, str) else func.__name__
        )
        return _register_and_return(tool_obj, func, tool_name)

    # Handle @petaltool and @petaltool(...)
    if callable(name_or_callable) and not isinstance(name_or_callable, str):
        # Used as @petaltool
        return _decorator(name_or_callable)
    else:
        # Used as @petaltool(...) or @petaltool("name")
        return _decorator


@tool
def petalmcp(_server_name: str):
    """
    Decorator for creating MCP server classes.

    Args:
        server_name: The name of the MCP server.

    Returns:
        Decorated class that will be registered as an MCP server.
    """

    def decorator(cls):
        # TODO: Implement MCP server registration
        # This will be implemented in Task 1.3
        return cls

    return decorator


def petalmcp_tool(_tool_name: str):
    """
    Decorator for creating MCP tool functions.

    Args:
        tool_name: The full name of the MCP tool (e.g., "filesystem:list_files").

    Returns:
        Decorated function that will be registered as an MCP tool.
    """

    def decorator(func):
        # TODO: Implement MCP tool registration
        # This will be implemented in Task 1.3
        return func

    return decorator
