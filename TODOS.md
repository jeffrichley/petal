# ✅ TODOS for Agent & Tool Framework Implementation

This list tracks the development tasks for the Chrona agent orchestration framework based on the `AGENT_API.md` specification.

---

## 🧠 AgentFactory

### Core Implementation

* [ ] Create `AgentFactory` class
* [ ] Implement `.add(step)` for function/step registration
* [ ] Add `.with_prompt()` for f-string prompt injection
* [ ] Add `.with_system_prompt()` to define system role
* [ ] Add `.with_chat()` to configure LLM support
* [ ] Add `.with_memory()` for memory support
* [ ] Add `.with_logger()` for internal state logging
* [ ] Add `.with_state()` for schema enforcement
* [ ] Add `.with_condition()` to guard step execution
* [ ] Add `.with_retry(n)` for retry logic
* [ ] Add `.with_timeout()` support
* [ ] Add `.with_tool_registry()` to inject tools
* [ ] Add `.with_mcp_proxy()` for deferred MCP toolkit resolution
* [ ] Add `.build()` to compile final agent Runnable
* [ ] Add `.run()` to execute with state dict
* [ ] Add `.as_node()` for LangGraph compatibility

### Extras

* [ ] Add `AgentFactory.from_config(path)`
* [ ] Add `AgentFactory.from_steps(*steps)`
* [ ] Add `.enable_dev_mode()`
* [ ] Add `.freeze()` to lock config
* [ ] Add `.set_logger()`

---

## 🛠 ToolFactory

### Core Implementation

* [ ] Create `ToolFactory` class
* [ ] Implement `.add(fn)` for direct registration
* [ ] Implement `.add_lazy(name, resolver)` for on-demand tools
* [ ] Implement `.resolve(name)` to retrieve tool
* [ ] Implement `.list()` to show all registered tools

### Discovery

* [ ] Add `.discover(from_folders)` for directory discovery
* [ ] Add `.discover(from_files)` for file-level loading
* [ ] Add `.from_folder(path)` shortcut
* [ ] Support `mcp:` style namespacing

---

## 🔁 GraphFactory

### Core Implementation

* [ ] Create `GraphFactory` class
* [ ] Add `.add_node(name, node)` to wire LangGraph nodes
* [ ] Add `.add_agent(name, agent_factory)` with `.as_node()` call
* [ ] Add `.connect(src, dest)` to define DAG edges
* [ ] Implement `.build()` to compile LangGraph
* [ ] Add `.run(initial_state)`

### Optional Features

* [ ] Add `.visualize()` for graph debugging
* [ ] Add `.expose_node(name)` to return node as `Runnable`
* [ ] Add `.from_yaml(path)` to parse graph structure
* [ ] Add `.from_registry()` to load known agents

---

## 🌐 Discovery & Config

* [ ] Add dynamic discovery defaults (e.g., `tools/`, `agents/`)
* [ ] Implement global `ToolRegistry.from_folder()` convenience helper
* [ ] Auto-register all `@tool_fn` decorated functions on import
* [ ] Enable agent discovery through annotated agent types

---

## 🧪 Testing & Dev Tools

* [ ] Unit tests for each method above
* [ ] Integration tests for full agent flow
* [ ] LangGraph test with multi-node chain
* [ ] YAML config roundtrip test
* [ ] Add pre-built test agents in `examples/`

---

End of task list.
