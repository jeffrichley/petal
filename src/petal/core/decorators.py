from typing import Any, Callable, Optional, TypeVar, Union, overload

from langchain.tools import tool
from langchain_core.tools import BaseTool

from petal.core.registry import ToolRegistry
from petal.core.tool_factory import ToolFactory

F = TypeVar("F", bound=Callable[..., Any])


def auto_namespace(func: Callable) -> str:
    """
    Generate an auto-namespaced name for a function.

    Args:
        func: The function to generate a name for

    Returns:
        The namespaced name in format "module:function_name"
        Falls back to just function_name if module is "__main__"
    """
    # If this is a LangChain BaseTool, get the original function if available
    original_func = getattr(func, "_original_func", func)
    module_name = (
        getattr(original_func, "__module__", None) or func.__module__ or "None"
    )
    func_name = getattr(original_func, "__name__", None) or func.__name__ or "unknown"

    # Fallback to just function name for __main__ module
    if module_name == "__main__":
        return func_name

    return f"{module_name}:{func_name}"


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
        # Always register under the namespaced name
        registry.add(tool_name, tool_obj)
        tool_obj._petal_registered = True  # type: ignore[attr-defined]
        tool_obj._original_func = func  # type: ignore[attr-defined]
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

        if isinstance(name_or_callable, str):
            # If the name already contains a colon, treat as fully qualified
            if ":" in name_or_callable:
                tool_name = name_or_callable
            else:
                # Auto-namespace the custom name
                original_func = getattr(func, "_original_func", func)
                module_name = (
                    getattr(original_func, "__module__", None) or func.__module__
                )
                if module_name == "__main__":
                    tool_name = name_or_callable
                else:
                    tool_name = f"{module_name}:{name_or_callable}"
            tool_obj = tool(tool_name, **tool_kwargs)(func)
            tool_obj.name = tool_name
        else:
            tool_obj = tool(**tool_kwargs)(func)
            tool_name = auto_namespace(func)
            tool_obj.name = tool_name

        return _register_and_return(tool_obj, func, tool_name)

    # Handle @petaltool and @petaltool(...)
    if callable(name_or_callable) and not isinstance(name_or_callable, str):
        # Used as @petaltool
        return _decorator(name_or_callable)
    else:
        # Used as @petaltool(...) or @petaltool("name")
        return _decorator


def petalmcp(server_name: str, config: dict):
    """
    Decorator for creating MCP server classes.
    Registers the server with ToolFactory.add_mcp using the official MCP client.
    """

    def decorator(cls):
        ToolFactory().add_mcp(server_name, mcp_config=config)
        return cls

    return decorator


def petalmcp_tool(tool_name: str):
    """
    Decorator for creating MCP tool functions.
    Registers the function as a tool under the mcp:server:tool namespace via ToolFactory.add.
    """

    def decorator(func):
        ToolFactory().add(tool_name, func)
        return func

    return decorator
