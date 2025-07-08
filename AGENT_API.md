# 🧠 Agent, Tool, and Graph Factory API (Chrona Framework)

This document outlines the full API specification for Chrona's core components:

* `AgentFactory`
* `ToolFactory`
* `GraphFactory`

Each factory supports a fluent, Pythonic interface with sensible defaults and advanced extensibility.

---

## 🧠 `AgentFactory` – Build an Agent Chain

### ✅ Purpose

Build and compose an agent from steps, tools, prompts, and runtime configuration. Produces a LangGraph-compatible `Runnable`.

### ✅ Constructor

```python
AgentFactory(name: str = None)
```

* `name`: Optional identifier for logging/discovery

---

### ✅ Fluent Methods

| Method                                         | Description                                         |                                            |
| ---------------------------------------------- | --------------------------------------------------- | ------------------------------------------ |
| \`.add(step: Callable                          | AgentStep)\`                                        | Add a step function or `@agent_step`.      |
| `.with_prompt(prompt: str)`                    | Injects a string prompt (f-string style).           |                                            |
| `.with_system_prompt(prompt: str)`             | Sets the persona/system message for LLM use.        |                                            |
| `.with_chat(llm: BaseChatModel = None)`        | Adds an LLM layer. If `None`, uses default factory. |                                            |
| `.with_memory(memory: BaseMemory)`             | Adds a memory backend (vector, chat, etc).          |                                            |
| `.with_react(toolset: ToolRegistry)`           | Adds ReAct pattern with tools.                      |                                            |
| `.with_logger()`                               | Logs each step into the state under `logs`.         |                                            |
| \`.with\_state(model: type\[TypedDict          | BaseModel])\`                                       | Enforces input/output schema across steps. |
| `.with_condition(fn: Callable)`                | Wraps downstream steps in a conditional check.      |                                            |
| `.with_retry(n: int)`                          | Retries failing steps.                              |                                            |
| `.with_timeout(seconds: float)`                | Aborts if timeout exceeded.                         |                                            |
| `.with_tool_registry(registry: ToolRegistry)`  | Binds a tool registry for use inside steps.         |                                            |
| `.with_mcp_proxy(name: str, loader: Callable)` | Registers MCP toolkit for on-demand tool loading.   |                                            |

---

### ✅ Execution

| Method                              | Description                                                      |
| ----------------------------------- | ---------------------------------------------------------------- |
| `.build() -> Runnable`              | Finalizes the agent. Returns a LangGraph-compatible callable.    |
| `.run(initial_state: dict) -> dict` | Executes immediately with provided input.                        |
| `.as_node() -> Runnable`            | Returns agent for use as LangGraph node (e.g., in GraphFactory). |

---

### ✅ Static Constructors

```python
AgentFactory.from_config(path: str)
AgentFactory.from_steps(*steps)
```

---

## 🛠 `ToolFactory` – Discover and Register Tools

### ✅ Purpose

Create and manage discoverable tool functions. Optionally supports lazy-resolution via MCP or plugin systems.

### ✅ Constructor

```python
ToolFactory(name: str = "default")
```

---

### ✅ Fluent Methods

| Method                                     | Description                                 |
| ------------------------------------------ | ------------------------------------------- |
| `.add(fn: Callable, name: str = None)`     | Register a tool manually.                   |
| `.add_lazy(name: str, resolver: Callable)` | Register a lazy-loaded proxy (for MCP/etc). |
| `.discover(from_folders: list[str])`       | Import tools from folders.                  |
| `.discover(from_files: list[str])`         | Import from specific files.                 |
| `.from_folder(path: str)`                  | Shortcut to load everything under a folder. |
| `.resolve(name: str) -> Callable`          | Get a callable by name.                     |
| `.list() -> list[str]`                     | Return all tool names.                      |

---

## 🔁 `GraphFactory` – Compose Multi-Agent Flows

### ✅ Purpose

Wire multiple agents (from `AgentFactory`) into a LangGraph-style DAG for advanced workflows.

### ✅ Constructor

```python
GraphFactory(name: str = None)
```

---

### ✅ Fluent Methods

| Method                                               | Description                                  |
| ---------------------------------------------------- | -------------------------------------------- |
| `.add_node(name: str, node: Runnable)`               | Add a LangGraph node by name.                |
| `.add_agent(name: str, agent_factory: AgentFactory)` | Auto-wires agent via `.as_node()`.           |
| `.connect(src: str, dest: str)`                      | Add edge from one node to another.           |
| `.from_yaml(path: str)`                              | Load structure from a YAML graph spec.       |
| `.from_registry()`                                   | Auto-wires known agents if registered.       |
| `.build() -> Graph`                                  | Returns a fully compiled LangGraph instance. |
| `.run(state: dict)`                                  | Executes entire graph with initial state.    |

---

### ✅ Output Helpers

| Method                    | Description                                   |
| ------------------------- | --------------------------------------------- |
| `.visualize()`            | Prints or returns a Mermaid graph.            |
| `.expose_node(name: str)` | Returns specific subgraph node as `Runnable`. |

---

## 🧪 Optional Shared Enhancements

All components can share:

* `.set_logger(fn: Callable)` — custom logger for debug output
* `.enable_dev_mode()` — adds extra verbose or validation info
* `.freeze()` — lock configuration to prevent runtime mutations

---

End of spec.
