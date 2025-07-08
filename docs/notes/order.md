All of your “Project Structure” items are already ✅ complete —so you can jump straight into the framework work. Here’s a revised, dependency-driven order:

---

## Phase 1 (Done)

**Project scaffolding & CI/CD**
*Everything under “Project Structure” is already checked off. *

---

## Phase 2: Core AgentFactory

Get your agent builder up and running before anything else depends on it.

1. Create the `AgentFactory` class
2. Implement `.add(step)`
3. Add `.with_prompt()`
4. Add `.with_system_prompt()`
5. Add `.with_chat()`
6. Add `.with_memory()`
7. Add `.with_logger()`
8. Add `.with_state()`
9. Add `.with_condition()`
10. Add `.with_retry(n)`
11. Add `.with_timeout()`
12. Add `.with_tool_registry()`
13. Add `.with_mcp_proxy()`
14. Add `.as_node()`
15. Add `.build()`
16. Add `.run()`

---

## Phase 3: Core ToolFactory

Once agents exist, you’ll need your tools resolved.

1. Create the `ToolFactory` class
2. Implement `.add(fn)`
3. Implement `.add_lazy(name, resolver)`
4. Implement `.resolve(name)`
5. Implement `.list()`

---

## Phase 4: Core GraphFactory

Wire agents and tools together into a LangGraph DAG.

1. Create the `GraphFactory` class
2. Add `.add_node(name, node)`
3. Add `.add_agent(name, agent_factory)`
4. Add `.connect(src, dest)`
5. Implement `.build()`
6. Add `.run(initial_state)`

---

## Phase 5: Extras & Dynamic Discovery

Layer on discoverability, configs, and convenience helpers.

* **AgentFactory extras** (`from_config`, `from_steps`, `enable_dev_mode`, etc.)
* **ToolFactory discovery** (`.discover()`, `from_folder`, MCP namespacing)
* **GraphFactory extras** (`.visualize()`, `.from_yaml()`, `.from_registry()`)
* **Global auto-registration** (default folders, `@tool_fn` auto-register, `ToolRegistry.from_folder()`)&#x20;

---

## Phase 6: Testing & QA

Lock it down with solid tests before your v0.1 release.

* Unit tests for every public method
* Integration tests of a small multi-node flow
* YAML config round-trip tests
* Example/demo in `examples/`
* Verify ≥ 80% coverage in CI badge

---

Let me know if you’d like to break any of these further into bite-sized sub-tasks!
