Great—now that the Core AgentFactory is in place, let’s move on to the **Core ToolFactory**. Here’s our next agenda:

1. **Define the `ToolFactory` class** in `src/petal/tool_factory.py`

   * Initialize an internal registry (e.g. `Dict[str, Callable]`) and a lazy-loader map.

2. **Implement `.add(fn: Callable, name: str | None = None) → ToolFactory`**

   * Register a tool by its function name (or an explicit `name`), storing the callable in the registry.

3. **Implement `.add_lazy(name: str, resolver: Callable[[], Callable]) → ToolFactory`**

   * Allow deferred registration of a tool via a zero-arg resolver that returns the callable when first requested.

4. **Implement `.resolve(name: str) → Callable`**

   * Look up the tool by name, loading it via the resolver if needed, and raise a clear `KeyError` if not found.

5. **Implement `.list() → List[str]`**

   * Return a sorted list of all registered tool names (including those available via lazy loaders).

---

### Example Usage

```python
from petal.tool_factory import ToolFactory

# 1) Create the factory
tools = ToolFactory()

# 2) Register an eager tool
def summarize(text: str) -> str:
    return text[:50] + "…"

tools.add(summarize)

# 3) Register a lazy tool
tools.add_lazy("translate", lambda: __import__("my_translator").translate)

# 4) Resolve and call
fn = tools.resolve("summarize")
print(fn("This is a very long text that needs shortening."))

fn2 = tools.resolve("translate")
print(fn2("bonjour", target="en"))

# 5) List available tools
print(tools.list())
# → ['summarize', 'translate']
```

Once you’ve got these five methods wired up with type hints, docstrings, and simple unit tests, we’ll hook AgentFactory into ToolFactory for full integration. Let me know when you’ve got the first draft!
