# ✅ TODOS for Agent & Tool Framework Implementation

This list tracks the development tasks for the Chrona agent orchestration framework based on the `AGENT_API.md` specification.

---

When working on these TODOS, refer to AGENT_API.md for specifications on how each piece of the framework should work.

## Project Structure
* [x] Add automatic documentation creation with Sphinx
* [x] Create a makefile
    - [x] Make a makefile target to run black and ruff to automatically fix issues
    - [x] Make a makefile target to run unit tests and generate a coverage report
* [x] Create an .env.example
    - [x] Add LangSmith environment variables for observability and tracing
* [x] Make a pre-commit hook for ensuring 80% test coverage
* [x] Add badges to README.md - https://github.com/jeffrichley/petal
    - [x] Build & CI - ![Build Status](https://github.com/jeffrichley/petal/actions/workflows/ci.yml/badge.svg)
    - [x] Test Coverage - ![Coverage](https://img.shields.io/codecov/c/gh/jeffrichley/petal)
    - [x] License - ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
    - [x] PyPI Version - ![PyPI version](https://badge.fury.io/py/petal.svg)
    - [x] Python Versions - ![Python Versions](https://img.shields.io/pypi/pyversions/petal.svg)
    - [x] pre-commit - ![pre-commit enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)
    - [x] Security - ![pip-audit](https://img.shields.io/badge/pip--audit-passing-brightgreen)
* [x] Update pyproject.toml’s [tool.setuptools.packages.find] (or equivalent) to point at src
* [x] Create a template project

## 🧠 AgentFactory

### Core Implementation

* [x] Create `AgentFactory` class
* [x] Implement `.add(step)` for function/step registration
* [x] Add `.with_prompt()` for f-string prompt injection
* [x] Add `.with_system_prompt()` to define system role
* [ ] Add `.with_chat()` to configure LLM support
* [ ] Add `.with_memory()` for memory support
* [ ] Add `.with_logger()` for internal state logging
* [ ] Add `.with_state()` for schema enforcement
* [ ] Add `.with_condition()` to guard step execution
* [ ] Add `.with_retry(n)` for retry logic
* [ ] Add `.with_timeout()` support
* [ ] Add `.with_tool_registry()` to inject tools
* [ ] Add `.with_mcp_proxy()` for deferred MCP toolkit resolution
* [x] Add `.build()` to compile final agent Runnable
* [x] Add `.run()` to execute with state dict
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

* [x] Create `ToolFactory` class
* [x] Implement `.add(fn)` for direct registration (now `.add(name, fn)`)
* [x] Implement `.resolve(name)` to retrieve tool
* [x] Implement `.list()` to show all registered tools
* [x] Implement `.add_mcp(name, resolver)` for mcp related tools

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

* [x] Unit tests for each method above
* [x] Integration tests for full agent flow
* [ ] LangGraph test with multi-node chain
* [ ] YAML config roundtrip test
* [x] Add pre-built test agents in `examples/`

---

## ✅ Completed During Development

### Infrastructure & Workflow
* [x] Fix Makefile workflow to prevent duplicate test runs
* [x] Add proper mypy configuration to exclude tests
* [x] Add troubleshooting section for corrupted coverage files in GIT_WORKFLOW.md
* [x] Achieve 100% test coverage across all modules
* [x] Implement comprehensive CI/CD pipeline with pre-commit hooks
* [x] Add development workflow documentation and rules

### Core Framework
* [x] Implement complete Petal framework with all core modules
* [x] Add configuration system with settings management
* [x] Add comprehensive type definitions and state management
* [x] Add example implementations for custom tools and agents
* [x] Add MCP server testing fixtures and integration

---

End of task list.
