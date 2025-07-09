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
* [x] Update pyproject.toml's [tool.setuptools.packages.find] (or equivalent) to point at src
* [x] Create a template project

## 🏗️ Factory.py Architectural Refactoring

### Phase 1: Extract Step Management (Strategy Pattern)

* [x] **Create StepStrategy Abstract Base Class**
  - [x] Create `src/petal/core/steps/__init__.py`
  - [x] Create `src/petal/core/steps/base.py` with `StepStrategy` ABC
  - [x] Define abstract methods: `create_step(config: Dict[str, Any]) -> Callable` and `get_node_name(index: int) -> str`
  - [x] Add type hints and comprehensive docstrings
  - [x] Create unit tests in `tests/petal/test_steps_base.py`

* [x] **Implement LLMStepStrategy**
  - [x] Create `src/petal/core/steps/llm.py` with `LLMStepStrategy` class
  - [x] Inherit from `StepStrategy` and implement abstract methods
  - [x] Move `LLMStep` class from `factory.py` to `steps/llm.py`
  - [x] Refactor `LLMStep` to use configuration object pattern
  - [x] Add comprehensive validation for LLM configuration
  - [x] Create unit tests in `tests/petal/test_steps_llm.py`
  - [x] Test all LLM provider configurations (OpenAI, etc.)

* [ ] **Implement CustomStepStrategy**
  - [ ] Create `src/petal/core/steps/custom.py` with `CustomStepStrategy` class
  - [ ] Support arbitrary callable functions as steps
  - [ ] Add validation for step function signatures
  - [ ] Support both sync and async functions
  - [ ] Create unit tests in `tests/petal/test_steps_custom.py`

* [x] **Create Step Registry**
  - [x] Create `src/petal/core/steps/registry.py` with `StepRegistry` class
  - [x] Implement `register(name: str, strategy: Type[StepStrategy])` method
  - [x] Implement `get_strategy(name: str) -> StepStrategy` method
  - [x] Add `_register_defaults()` method to register built-in strategies
  - [x] Add validation and error handling for unknown step types
  - [x] Create unit tests in `tests/petal/test_steps_registry.py`

### Phase 2: Configuration Management (Configuration Object Pattern)

* [x] **Create AgentConfig Data Class**
  - [x] Create `src/petal/core/config/__init__.py`
  - [x] Create `src/petal/core/config/agent.py` with `AgentConfig` dataclass
  - [x] Define fields: `state_type`, `steps`, `memory`, `graph_config`
  - [x] Add `add_step(strategy: StepStrategy, config: Dict[str, Any])` method
  - [x] Add `set_memory(memory_config: Dict[str, Any])` method
  - [x] Add validation methods for configuration integrity
  - [x] Create unit tests in `tests/petal/test_config_agent.py`

* [ ] **Create State Type Factory**
  - [ ] Create `src/petal/core/config/state.py` with `StateTypeFactory` class
  - [ ] Move `_create_state_type()` logic from `AgentFactory` to this class
  - [ ] Implement `create_with_messages(base_type: type) -> type` static method
  - [ ] Implement `create_mergeable(base_type: type) -> type` static method
  - [ ] Add caching mechanism for dynamic type creation
  - [ ] Add comprehensive error handling for type creation failures
  - [ ] Create unit tests in `tests/petal/test_config_state.py`

* [ ] **Create Graph Configuration**
  - [ ] Create `src/petal/core/config/graph.py` with `GraphConfig` class
  - [ ] Define graph building parameters and edge configurations
  - [ ] Support different graph topologies (linear, branching, etc.)
  - [ ] Add validation for graph structure integrity
  - [ ] Create unit tests in `tests/petal/test_config_graph.py`

### Phase 3: Builder Pattern with Composition

* [ ] **Create AgentBuilder Class**
  - [ ] Create `src/petal/core/builders/__init__.py`
  - [ ] Create `src/petal/core/builders/agent.py` with `AgentBuilder` class
  - [ ] Implement fluent interface with `with_step()`, `with_memory()`, etc.
  - [ ] Use composition with `AgentConfig` and `StepRegistry`
  - [ ] Add validation for builder state consistency
  - [ ] Create unit tests in `tests/petal/test_builders_agent.py`

* [ ] **Create AgentBuilderDirector**
  - [ ] Create `src/petal/core/builders/director.py` with `AgentBuilderDirector` class
  - [ ] Move complex building logic from `AgentFactory.build()` to this class
  - [ ] Implement `build() -> Agent` method
  - [ ] Add `_create_state_type()` and `_build_graph()` private methods
  - [ ] Add comprehensive error handling for build failures
  - [ ] Create unit tests in `tests/petal/test_builders_director.py`

* [ ] **Create Step Configuration Handlers**
  - [ ] Create `src/petal/core/builders/handlers/__init__.py`
  - [ ] Create `src/petal/core/builders/handlers/base.py` with `StepConfigHandler` ABC
  - [ ] Implement Chain of Responsibility pattern for step configuration
  - [ ] Create `src/petal/core/builders/handlers/llm.py` with `LLMConfigHandler`
  - [ ] Create `src/petal/core/builders/handlers/custom.py` with `CustomConfigHandler`
  - [ ] Add comprehensive error handling and validation
  - [ ] Create unit tests in `tests/petal/test_builders_handlers.py`

### Phase 4: Refactor Existing Factory

* [ ] **Update AgentFactory to Use New Architecture**
  - [ ] Modify `src/petal/core/factory.py` to use new builder pattern
  - [ ] Replace direct step management with `AgentBuilder` composition
  - [ ] Update `with_chat()` method to use new step registry
  - [ ] Update `add()` method to use new step strategies
  - [ ] Maintain backward compatibility during transition
  - [ ] Update all existing tests to work with new architecture

* [ ] **Remove ChatStepBuilder**
  - [ ] Deprecate `ChatStepBuilder` class in favor of new builder pattern
  - [ ] Update all examples and documentation to use new approach
  - [ ] Remove `ChatStepBuilder` after migration is complete
  - [ ] Update tests to reflect new architecture

* [ ] **Update State Management**
  - [ ] Replace dynamic type creation in `AgentFactory` with `StateTypeFactory`
  - [ ] Update state type caching mechanism
  - [ ] Add better error messages for state type creation failures
  - [ ] Update tests to use new state management approach

### Phase 5: Extensibility and Advanced Features

* [ ] **Add Plugin System for Step Types**
  - [ ] Create `src/petal/core/plugins/__init__.py`
  - [ ] Create `src/petal/core/plugins/base.py` with plugin interface
  - [ ] Implement automatic discovery of step type plugins
  - [ ] Add plugin registration and management system
  - [ ] Create example plugins for common step types
  - [ ] Create unit tests in `tests/petal/test_plugins.py`

* [ ] **Add Configuration Validation**
  - [ ] Create `src/petal/core/validation/__init__.py`
  - [ ] Create `src/petal/core/validation/config.py` with validation schemas
  - [ ] Use Pydantic for configuration validation
  - [ ] Add comprehensive validation for all configuration objects
  - [ ] Create unit tests in `tests/petal/test_validation.py`

* [ ] **Add Advanced Graph Building**
  - [ ] Create `src/petal/core/graph/__init__.py`
  - [ ] Create `src/petal/core/graph/builder.py` with advanced graph building
  - [ ] Support conditional edges and branching logic
  - [ ] Support parallel execution paths
  - [ ] Add graph visualization capabilities
  - [ ] Create unit tests in `tests/petal/test_graph.py`

### Phase 6: Testing and Documentation

* [ ] **Comprehensive Integration Testing**
  - [ ] Create `tests/integration/test_factory_refactor.py`
  - [ ] Test complete agent building workflows with new architecture
  - [ ] Test backward compatibility with existing code
  - [ ] Test performance impact of new architecture
  - [ ] Test error handling and edge cases

* [ ] **Update Documentation**
  - [ ] Update `docs/source/api/factory.rst` with new architecture
  - [ ] Create migration guide from old to new factory usage
  - [ ] Update all examples to use new builder pattern
  - [ ] Add architectural decision records (ADRs) for the refactoring
  - [ ] Update README.md with new usage patterns

* [ ] **Performance Optimization**
  - [ ] Profile new architecture for performance bottlenecks
  - [ ] Optimize step creation and configuration
  - [ ] Optimize state type creation and caching
  - [ ] Add performance benchmarks and monitoring
  - [ ] Create performance regression tests

### Phase 7: Cleanup and Finalization

* [ ] **Remove Deprecated Code**
  - [ ] Remove old `ChatStepBuilder` class completely
  - [ ] Remove old step management code from `AgentFactory`
  - [ ] Remove old state type creation logic
  - [ ] Clean up unused imports and dependencies
  - [ ] Update type hints and annotations

* [ ] **Final Testing and Validation**
  - [ ] Run complete test suite to ensure no regressions
  - [ ] Test all existing examples and playground code
  - [ ] Validate that all existing functionality still works
  - [ ] Check code coverage is maintained or improved
  - [ ] Run performance benchmarks to ensure no degradation

* [ ] **Update Development Workflow**
  - [ ] Update `PLANNING.md` with new architecture decisions
  - [ ] Update `GIT_WORKFLOW.md` if needed for new patterns
  - [ ] Update pre-commit hooks if needed for new code structure
  - [ ] Update CI/CD pipeline if needed for new testing patterns

---

## 🧠 AgentFactory

### Core Implementation

* [x] Create `AgentFactory` class
* [x] Implement `.add(step)` for function/step registration
* [x] Add `.with_prompt()` for f-string prompt injection
* [x] Add `.with_system_prompt()` to define system role
* [x] Add `.with_chat()` to configure LLM support
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

### AgentFactory Enhancements
* [x] Refactor AgentFactory to use LangGraph RunnableSequence for step chaining
* [x] Implement per-LLM-step prompt configuration with ChatStepBuilder
* [x] Add support for multiple LLM steps with different prompts in same chain
* [x] Implement automatic LLM invocation during agent execution
* [x] Add both sync and async execution support
* [x] Set default model to 'gpt-4o-mini' and temperature to 0
* [x] Add comprehensive tests for new chaining behavior
* [x] Create playground example demonstrating multi-step LLM chains

---

## ✅ Recently Completed (2024-12-19)

### Phase 1: Step Management (Strategy Pattern) ✅
- **Completed:** Full step management system with strategy pattern
- **Features:** StepStrategy ABC, LLMStepStrategy, StepRegistry
- **Coverage:** Comprehensive test coverage for all step components
- **Status:** Ready for integration with AgentFactory refactoring

### Task 1.4: Create AgentConfig ✅
- **Completed:** Full AgentConfig implementation with Pydantic models
- **Features:** StepConfig, MemoryConfig, GraphConfig, LLMConfig, LoggingConfig
- **Coverage:** 96% test coverage with 28 comprehensive tests
- **Quality:** All mypy, linting, and formatting checks pass
- **Status:** Ready for integration with AgentFactory refactoring

---

End of task list.
