# 📐 PLANNING.md — Chrona Agent & Tool Orchestration Framework

This document outlines the design philosophy, architectural principles, and implementation strategy for the Chrona agent orchestration system. The goal is to create an ergonomic, extensible framework for defining, composing, and executing LLM-powered agents using LangChain, LangGraph, and Pydantic.

---

## 🎯 Project Goals

1. **Agent Chaining via Python API**

   * Define agents and tool-using workflows using expressive builder chains (e.g., `.with_prompt().with_chat().with_logger()`)
   * Hide low-level LangGraph + LangChain complexity from end users

2. **Zero-Config Agent Discovery**

   * Auto-register tools from `tools/` and agents from `agents/`
   * Allow for project-specific overrides and dynamic tool injection (e.g., MCP-based)

3. **Composable, Modular Architecture**

   * Agents should be portable as nodes within LangGraph
   * Tools should be lazy-loadable, composable, and traceable

4. **Typed State Handling**

   * Use `TypedDict` or `BaseModel` to define agent input/output schema
   * Enable type-checked prompt formatting and chaining

5. **LangGraph as Runtime Backbone**

   * Use LangGraph internally, but abstract away from user
   * Provide `.run()` and `.as_node()` methods on all factories

---

## 🏗 Architecture

### AgentFactory

* Encapsulates all metadata and steps for a single LLM-powered agent
* Methods are chainable to add:

  * Prompts
  * System instructions
  * Tool registries
  * Retry/timeouts/memory/logging wrappers
* Compiles into LangGraph node or full graph if `.build()` is used

### ToolFactory

* Discovers, registers, and resolves callable tools
* Supports lazy registration and dynamic toolkits (e.g. MCP proxies)
* Provides `resolve()`, `list()`, and `from_folder()` methods

### GraphFactory

* Wires agents and steps into complete LangGraph workflows
* Allows composition of `AgentFactory` objects via `.add_agent()`
* Supports `.connect()` edge declarations

---

## 🧱 Key Components

### Agent Step

```python
@agent_step
def inspire(state):
    mood = state['mood']
    return {'quote': get_inspiration(mood)}
```

Each step receives and returns a state dictionary. Decorator adds LangGraph-compatible wrapping + metadata.

### Tool Function

```python
@tool_fn
def get_inspiration(mood: str) -> str:
    ...
```

Decorated tools are auto-registered. Support for MCP-style resolution with `mcpname:toolname` syntax.

### Full Agent Example

```python
AgentFactory()
  .add(inspire)
  .with_prompt("Inspire someone feeling {mood}")
  .with_chat()
  .with_tool_registry(MyToolRegistry)
  .build()
```

---

## ⚙ Design Principles

* **Pythonic**: Users should compose agents via fluent chains — not YAML-first or config-only
* **Discoverable**: All tools and agents must support auto-discovery
* **LLM-Flexible**: Support per-agent LLM bindings via LLM Factory
* **LangGraph-Compatible**: All agents and graphs expose `.as_node()` and `.run()`
* **Safe Defaults**: Fallback to `tools/` and `agents/` folders when nothing specified

---

## 📊 Milestones

1. **Week 1**:

   * ToolFactory + `@tool_fn` + discovery
   * AgentFactory: base + prompt/chat/logger/step
2. **Week 2**:

   * State validation + retry/timeout/memory
   * GraphFactory + LangGraph runtime
3. **Week 3**:

   * YAML config loader + test graph
   * MCP tool proxy
   * Devtools for logging, visualization
4. **Week 4**:

   * Polishing, scaffolded examples, test cases
   * Real-world agent flow (Chrona shorts generator)

---

## 🧪 Test Plan

* Unit test for every method in each factory
* YAML-to-agent config roundtrip
* Tool loading and resolution test
* LangGraph node test via `.as_node()`
* End-to-end short-form video agent test

---

## ✨ Inspiration

* LangChain
* LangGraph
* FastAPI
* React/Fluent UI pattern chaining
* pydantic-ai + MCP-style toolkits

---

This document governs implementation, testing, and team collaboration for the Chrona framework. Keep it up to date as we evolve.
