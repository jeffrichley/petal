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
    - [x] Python Versions - ![Python Versions](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
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

* [x] **Implement CustomStepStrategy** ✅ (Completed 2024-06-22)
  - [x] Create `src/petal/core/steps/custom.py` with `CustomStepStrategy` class
  - [x] Support arbitrary callable functions as steps
  - [x] Add validation for step function signatures
  - [x] Support both sync and async functions
  - [x] Create unit tests in `tests/petal/test_steps_custom.py` (100% coverage)

* [x] **Create Step Registry**
  - [x] Create `src/petal/core/steps/registry.py` with `StepRegistry` class
  - [x] Implement `register(name: str, strategy: Type[StepStrategy])` method
  - [x] Implement `get_strategy(name: str) -> StepStrategy` method
  - [x] Add `_register_defaults()` method to register built-in strategies
  - [x] Add validation and error handling for unknown step types
  - [x] Create unit tests in `tests/petal/test_steps_registry.py`

* [x] **Implement ToolStepStrategy** ✅ (Completed 2024-12-22)
  - [x] Create `src/petal/core/steps/tool.py` with `ToolStepStrategy` class
  - [x] Support LangChain tools and ReAct-style scratchpad
  - [x] Implement conditional routing and tool execution
  - [x] Create unit tests in `tests/petal/test_steps_tool.py` (100% coverage)

* [x] **Implement ReactStepStrategy** ✅ (Completed 2024-12-22)
  - [x] Create `src/petal/core/steps/react.py` with `ReactStepStrategy` class
  - [x] Support ReAct loop with tool usage and reasoning
  - [x] Implement max_iterations and reasoning prompts
  - [x] Create unit tests in `tests/petal/test_steps_react.py` (100% coverage)

### Phase 2: Configuration Management (Configuration Object Pattern)

* [x] **Create AgentConfig Data Class**
  - [x] Create `src/petal/core/config/__init__.py`
  - [x] Create `src/petal/core/config/agent.py` with `AgentConfig` dataclass
  - [x] Define fields: `state_type`, `steps`, `memory`, `graph_config`
  - [x] Add `add_step(strategy: StepStrategy, config: Dict[str, Any])` method
  - [x] Add `set_memory(memory_config: Dict[str, Any])` method
  - [x] Add validation methods for configuration integrity
  - [x] Create unit tests in `tests/petal/test_config_agent.py`

* [x] **Create State Type Factory**
  - [x] Create `src/petal/core/config/state.py` with `StateTypeFactory` class
  - [x] Move `_create_state_type()` logic from `AgentFactory` to this class
  - [x] Implement `create_with_messages(base_type: type) -> type` static method
  - [x] Implement `create_mergeable(base_type: type) -> type` static method
  - [x] Add caching mechanism for dynamic type creation
  - [x] Add comprehensive error handling for type creation failures
  - [x] Create unit tests in `tests/petal/test_config_state.py`

* [x] **Create Graph Configuration** ✅ (Implemented in AgentConfig)
  - [x] Create `src/petal/core/config/agent.py` with `GraphConfig` class
  - [x] Define graph building parameters and edge configurations
  - [x] Support different graph topologies (linear, branching, etc.)
  - [x] Add validation for graph structure integrity
  - [x] Unit tests included in `tests/petal/test_config_agent.py`

### Phase 3: Builder Pattern with Composition

* [x] **Create AgentBuilder Class**
  - [x] Create `src/petal/core/builders/__init__.py`
  - [x] Create `src/petal/core/builders/agent.py` with `AgentBuilder` class
  - [x] Implement fluent interface with `with_step()`, `with_memory()`, etc.
  - [x] Use composition with `AgentConfig` and `StepRegistry`
  - [x] Add validation for builder state consistency
  - [x] Create unit tests in `tests/petal/test_builders_agent.py`

* [x] **Create AgentBuilderDirector** ✅ (Completed 2024-06-22)
  - [x] Create `src/petal/core/builders/director.py` with `AgentBuilderDirector` class
  - [x] Move complex building logic from `AgentFactory.build()` to this class
  - [x] Implement `build() -> Agent` method
  - [x] Add `_create_state_type()` and `_build_graph()` private methods
  - [x] Add `_validate_configuration()` method
  - [x] Add comprehensive error handling for build failures
  - [x] Create unit tests in `tests/petal/test_builders_director.py`
  - [x] Integration with AgentBuilder complete
  - [x] All tests passing with 100% coverage

* [x] **Create Step Configuration Handlers** ✅ (Completed 2024-06-22)
  - [x] Create `src/petal/core/builders/handlers/__init__.py`
  - [x] Create `src/petal/core/builders/handlers/base.py` with `StepConfigHandler` ABC
  - [x] Implement Chain of Responsibility pattern for step configuration
  - [x] Create `src/petal/core/builders/handlers/llm.py` with `LLMConfigHandler`
  - [x] Create `src/petal/core/builders/handlers/custom.py` with `CustomConfigHandler`
  - [x] Add comprehensive error handling and validation
  - [x] Create unit tests in `tests/petal/test_builders_handlers.py`

### Phase 4: Refactor Existing Factory

* [x] **Update AgentFactory to Use New Architecture** ✅ (Completed 2024-06-22)
  - [x] Modify `src/petal/core/factory.py` to use new builder pattern
  - [x] Replace direct step management with `AgentBuilder` composition
  - [x] Update `with_chat()` method to use new step registry
  - [x] Update `add()` method to use new step strategies
  - [x] Maintain backward compatibility during transition
  - [x] Update all existing tests to work with new architecture

* [x] **Remove ChatStepBuilder** ✅ (Completed 2024-06-22)
  - [x] Remove `ChatStepBuilder` class completely from `factory.py`
  - [x] Update all examples and documentation to use new approach
  - [x] Update tests to reflect new architecture
  - [x] All functionality preserved through new fluent interface

* [x] **Update State Management**
  - [x] Replace dynamic type creation in `AgentFactory` with `StateTypeFactory`
  - [x] Update state type caching mechanism
  - [x] Add better error messages for state type creation failures
  - [x] Update tests to use new state management approach

### Phase 5: Extensibility and Advanced Features

* [x] **Add Plugin System for Step Types** (Completed 2024-06-22)
  - [x] Create `src/petal/core/plugins/__init__.py`
  - [x] Create `src/petal/core/plugins/base.py` with plugin interface
  - [x] Implement automatic discovery of step type plugins
  - [x] Add plugin registration and management system
  - [x] Create example plugins for common step types
  - [x] Create unit tests in `tests/petal/test_plugins.py`

* [x] **Add Configuration Validation**
  - [x] Create `src/petal/core/validation/__init__.py`
  - [x] Create `src/petal/core/validation/config.py` with validation schemas
  - [x] Use Pydantic for configuration validation
  - [x] Add comprehensive validation for all configuration objects
  - [x] Create unit tests in `tests/petal/test_validation.py`

* [ ] **Add Advanced Graph Building**
  - [ ] Create `src/petal/core/graph/__init__.py`
  - [ ] Create `src/petal/core/graph/builder.py` with advanced graph building
  - [ ] Support conditional edges and branching logic
  - [ ] Support parallel execution paths
  - [ ] Add graph visualization capabilities
  - [ ] Create unit tests in `tests/petal/test_graph.py`

### Phase 6: Testing and Documentation

* [x] **Comprehensive Integration Testing**
  - [x] Create `tests/integration/test_yaml_loading.py`
  - [x] Test complete agent building workflows with new architecture
  - [x] Test backward compatibility with existing code
  - [x] Test performance impact of new architecture
  - [x] Test error handling and edge cases

* [x] **Update Documentation**
  - [x] Update `docs/source/api/factory.rst` with new architecture
  - [x] Create migration guide from old to new factory usage
  - [x] Update all examples to use new builder pattern
  - [x] Add architectural decision records (ADRs) for the refactoring
  - [x] Update README.md with new usage patterns

* [ ] **Performance Optimization**
  - [ ] Profile new architecture for performance bottlenecks
  - [ ] Optimize step creation and configuration
  - [ ] Optimize state type creation and caching
  - [ ] Add performance benchmarks and monitoring
  - [ ] Create performance regression tests

## 🎯 YAML Node Loading Implementation

### Phase 1: Core YAML Loading Infrastructure

* [x] **Create YAML Configuration Models** ✅ (Completed 2024-12-19)
  - [x] Create `src/petal/core/config/yaml.py` with Pydantic models
  - [x] Define `BaseNodeConfig` with common fields (type, name, description, enabled)
  - [x] Define `LLMNodeConfig` with provider, model, temperature, max_tokens, prompt, system_prompt
  - [x] Define `ReactNodeConfig` with tools, reasoning_prompt, system_prompt, max_iterations
  - [x] Add comprehensive validation and error handling
  - [x] Create unit tests in `tests/petal/test_config_yaml.py` (100% coverage)
  - [x] All 39 tests passing with comprehensive validation
  - [x] No regressions in existing functionality

* [x] **Create YAML Parser and Type Detection** ✅ (Completed 2024-12-19)
  - [x] Create `src/petal/core/yaml/__init__.py`
  - [x] Create `src/petal/core/yaml/parser.py` with `YAMLNodeParser` class
  - [x] Implement `parse_node_config(path: str) -> BaseNodeConfig` method
  - [x] Implement `detect_node_type(yaml_data: Dict) -> str` method
  - [x] Add support for YAML validation and schema checking
  - [x] Create unit tests in `tests/petal/test_yaml_parser.py`

* [x] **Extend AgentFactory with YAML Support** ✅ (Completed 2024-12-19)
  - [x] Add `node_from_yaml(path: str) -> Runnable` method to AgentFactory
  - [x] Integrate with existing step registry and strategies
  - [x] Add validation for YAML file existence and format
  - [x] Add error handling for malformed YAML configurations
  - [x] Create unit tests in `tests/petal/test_factory.py`

* [x] **Create Node Configuration Handlers** ✅ (Completed 2024-12-19)
  - [x] Create `src/petal/core/yaml/handlers/__init__.py`
  - [x] Create `src/petal/core/yaml/handlers/base.py` with `NodeConfigHandler` ABC
  - [x] Create `src/petal/core/yaml/handlers/llm.py` with `LLMNodeHandler`
  - [x] Create `src/petal/core/yaml/handlers/react.py` with `ReactNodeHandler`
  - [x] Create `src/petal/core/yaml/handlers/custom.py` with `CustomNodeHandler`
  - [x] Create `src/petal/core/yaml/handlers/factory.py` with `NodeHandlerFactory`
  - [x] Implement `create_node(config: BaseNodeConfig) -> Runnable` method
  - [x] Create unit tests in `tests/petal/test_yaml_handlers.py`

### Phase 2: Advanced Configuration Support

* [x] **Add State Schema Support** ✅ (Completed 2024-12-19)
  - [x] Extend YAML models to support state schema definitions
  - [x] Add `StateSchemaConfig` with field definitions and validation
  - [x] Integrate with existing `StateTypeFactory`
  - [x] Support dynamic state type creation from YAML
  - [x] Create unit tests for state schema loading

* [x] **Add Tool Registry Integration** ✅ (Completed 2024-12-19)
  - [x] Extend `ReactNodeConfig` to support tool references
  - [x] Add tool discovery and resolution from YAML
  - [x] Support both direct tool names and tool configurations
  - [x] Integrate with existing `ToolFactory`
  - [x] Create unit tests for tool integration

* [ ] **Add Memory Configuration**
  - [ ] Extend YAML models to support memory configuration
  - [ ] Add `MemoryConfig` with persistence settings
  - [ ] Support different memory backends (file, database, etc.)
  - [ ] Integrate with existing memory management
  - [ ] Create unit tests for memory configuration

### Phase 3: Logging and Debugging

* [ ] **Add Logging Configuration**
  - [ ] Extend YAML models to support logging settings
  - [ ] Add `LoggingConfig` with level, format, handlers
  - [ ] Support Rich logging integration
  - [ ] Add debug mode configuration
  - [ ] Create unit tests for logging configuration

* [ ] **Add Validation and Error Handling**
  - [ ] Add comprehensive YAML schema validation
  - [ ] Add detailed error messages for configuration issues
  - [ ] Add configuration validation before node creation
  - [ ] Add support for configuration inheritance and defaults
  - [ ] Create unit tests for validation and error handling

### Phase 4: Integration and Testing

* [x] **Create Example YAML Configurations** ✅ (Completed 2024-12-19)
  - [x] Create `examples/yaml/llm_node.yaml` with LLM configuration
  - [x] Create `examples/yaml/react_node.yaml` with React configuration
  - [x] Create `examples/yaml/custom_node.yaml` with custom configuration
  - [x] Create `examples/yaml/complex_node.yaml` with all features
  - [x] Add documentation for YAML format and options

* [x] **Integration Testing** ✅ (Completed 2024-12-19)
  - [x] Create `tests/integration/test_yaml_loading.py`
  - [x] Test complete node creation from YAML files
  - [x] Test error handling for invalid configurations
  - [x] Test integration with existing AgentFactory methods
  - [x] Test performance of YAML loading vs programmatic creation

* [x] **Documentation and Examples** ✅ (Completed 2024-12-19)
  - [x] Update `docs/source/api/factory.rst` with YAML loading
  - [x] Create YAML configuration guide
  - [x] Add examples to playground and demo files
  - [x] Create migration guide from programmatic to YAML configuration
  - [x] Add troubleshooting section for common YAML issues

### Phase 5: Optimization and Polish

* [ ] **Performance Optimization**
  - [ ] Add YAML parsing caching for repeated loads
  - [ ] Optimize node creation from YAML configurations
  - [ ] Add lazy loading for large configurations
  - [ ] Create performance benchmarks
  - [ ] Add memory usage monitoring

* [ ] **Advanced Features**
  - [ ] Add support for YAML includes and inheritance
  - [ ] Add support for environment variable substitution
  - [ ] Add support for conditional configuration based on environment
  - [ ] Add support for configuration templates and macros
  - [ ] Create unit tests for advanced features

### Phase 6: Final Validation

* [x] **Comprehensive Testing** ✅ (Completed 2024-12-19)
  - [x] Run complete test suite to ensure no regressions
  - [x] Test all existing examples with new YAML loading
  - [x] Validate that all existing functionality still works
  - [x] Check code coverage is maintained or improved
  - [x] Run performance benchmarks to ensure no degradation

* [x] **Documentation** ✅ (Completed 2024-12-19)
  - [x] Update README.md with YAML loading examples
  - [x] Create migration guide for existing users
  - [x] Add architectural decision records (ADRs) for YAML support
  - [x] Update CI/CD pipeline for YAML configuration testing

## 🧠 AgentFactory

### Core Implementation

* [x] Create `AgentFactory` class
* [x] Implement `.add(step)` for function/step registration
* [x] Add `.with_prompt()` for f-string prompt injection
* [x] Add `.with_system_prompt()` to define system role
* [x] Add `.with_chat()` to configure LLM support
* [ ] Add `.with_memory()` for memory support
* [ ] Add `.with_logger()` for internal state logging
* [x] Add `.with_state()` for schema enforcement
* [ ] Add `.with_condition()` to guard step execution
* [ ] Add `.with_retry(n)` for retry logic
* [ ] Add `.with_timeout()` support
* [x] Add `.with_tool_registry()` to inject tools
* [x] Add `.with_mcp_proxy()` for deferred MCP toolkit resolution
* [x] Add `.build()` to compile final agent Runnable
* [x] Add `.run()` to execute with state dict
* [x] Add `.as_node()` for LangGraph compatibility

### Extras

* [x] Add `AgentFactory.node_from_yaml(path)` to load nodes from YAML with automatic type detection
* [ ] Add `AgentFactory.from_config(path)`
* [ ] Add `AgentFactory.from_steps(*steps)`
* [ ] Add `.enable_dev_mode()`
* [ ] Add `.freeze()` to lock config
* [ ] Add `.set_logger()`

---

## 🛠 ToolFactory

### Core Implementation

* [x] Create `ToolFactory` class
* [x] Implement `.add(name, fn)` for direct registration
* [x] Implement `.resolve(name)` to retrieve tool
* [x] Implement `.list()` to show all registered tools
* [x] Implement `.add_mcp(name, resolver)` for mcp related tools

### Discovery

* [ ] Add `.discover(from_folders)` for directory discovery
* [ ] Add `.discover(from_files)` for file-level loading
* [ ] Add `.from_folder(path)` shortcut
* [x] Support `mcp:` style namespacing

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
* [x] YAML config roundtrip test
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

## ✅ Recently Completed (2024-12-22)

### Phase 1: Step Management (Strategy Pattern) ✅
- **Completed:** Full step management system with strategy pattern
- **Features:** StepStrategy ABC, LLMStepStrategy, StepRegistry
- **Coverage:** Comprehensive test coverage for all step components
- **Status:** Ready for integration with AgentFactory refactoring

### Task 1.4: Create AgentConfig ✅
- **Completed:** Full AgentConfig implementation with Pydantic models
- **Features:** StepConfig, MemoryConfig, GraphConfig, LLMConfig, LoggingConfig
- **Coverage:** 100% test coverage with 28 comprehensive tests
- **Quality:** All mypy, linting, and formatting checks pass
- **Status:** Ready for integration with AgentFactory refactoring

### Task 1.5: Create StateTypeFactory ✅
- **Completed:** Full StateTypeFactory implementation with caching
- **Features:** create_with_messages, create_mergeable, caching mechanism
- **Coverage:** 100% test coverage with comprehensive tests
- **Quality:** All mypy, linting, and formatting checks pass
- **Status:** Ready for integration with AgentFactory refactoring

### Task 1.6: Create AgentBuilder ✅ (Completed 2024-06-21, 100% coverage, all tests passing)
- **Completed:** Full AgentBuilder implementation with fluent interface
- **Features:** with_step, with_memory, with_llm, with_logging, with_graph_config, build
- **Coverage:** 100% test coverage with comprehensive tests
- **Quality:** All mypy, linting, and formatting checks pass
- **Status:** Integration with AgentBuilderDirector complete

### Task 1.7: Create AgentBuilderDirector ✅ (Completed 2024-12-22, 100% coverage, all tests passing)
- **Completed:** Full AgentBuilderDirector implementation
- **Features:** build, _create_state_type, _build_graph, _validate_configuration
- **Coverage:** 100% test coverage with comprehensive tests
- **Quality:** All mypy, linting, and formatting checks pass
- **Status:** Integration with AgentBuilder complete, MCP compliance ensured

### Task 2.1: Update AgentFactory to Use New Architecture ✅ (Completed 2024-06-22)
- **Completed:** AgentFactory now uses new architecture internally
- **Features:** Backward compatibility maintained, AgentBuilder integration
- **Coverage:** All existing tests pass, new architecture test added
- **Quality:** All mypy, linting, and formatting checks pass
- **Status:** Full integration complete, ChatStepBuilder still exists for compatibility

### Task 2.2: Add CustomStepStrategy ✅ (Completed 2024-06-22)
- **Completed:** CustomStepStrategy implemented and registered
- **Features:** Support for arbitrary callable functions as steps
- **Coverage:** Strategy implemented, 100% test coverage
- **Quality:** All mypy, linting, and formatting checks pass
- **Status:** Ready for use

### Task 3.1: Remove ChatStepBuilder ✅ (Completed 2024-06-22)
- **Completed:** ChatStepBuilder completely removed from codebase
- **Features:** New fluent interface with direct AgentFactory methods
- **Coverage:** 100% test coverage with comprehensive error case tests
- **Quality:** All mypy, linting, and formatting checks pass
- **Status:** Complete migration to new architecture

### API Improvements and Enhancements ✅ (Completed 2024-12-22)
- **Completed:** Improved `with_llm` method to use named parameters instead of config dict
- **Features:** `with_llm(provider: str, model: str, temperature: float = 0.0, max_tokens: int = 1000)`
- **Coverage:** Updated playground examples and tests to use new API
- **Quality:** More Pythonic and type-safe API design
- **Status:** Backward compatibility maintained, new API preferred

### System Prompt Handling ✅ (Completed 2024-12-22)
- **Completed:** Enhanced system prompt support with state variable formatting
- **Features:** `with_system_prompt()` method, automatic state variable interpolation
- **Coverage:** Fixed `_build_llm_messages` to format system prompts with state
- **Quality:** Proper error handling for missing keys in system prompt templates
- **Status:** Full integration with LLM steps and state management

### Rich Logging Integration ✅ (Completed 2024-12-22)
- **Completed:** Enhanced playground2.py with Rich logging and visual improvements
- **Features:** Rich panels, tables, spinners, colors, and proper markup handling
- **Coverage:** Fixed logging issues, added timing info, improved results display
- **Quality:** Beautiful and informative console output with proper error handling
- **Status:** Ready for production use with comprehensive logging

### Error Handling and Testing ✅ (Completed 2024-12-22)
- **Completed:** Added comprehensive error handling for missing keys in system prompts
- **Features:** Test coverage for LLMStep error handling when system prompt references missing keys
- **Coverage:** Async test in `test_steps_llm.py` with proper error message validation
- **Quality:** Robust error handling with descriptive error messages
- **Status:** Production-ready error handling with full test coverage

### React Loop Testing and Tool Conversion ✅ (Completed 2024-12-22)
- **Completed:** Fixed TypedDict instance checking issues in React loop tests
- **Features:** Replaced isinstance checks with dict-like validation for TypedDict compatibility
- **Coverage:** Fixed all React loop tests and added comprehensive tool conversion tests
- **Quality:** All tests passing, proper mocking of LLM with_structured_output method
- **Status:** React loop fully functional with proper type handling

### Playground and Code Quality Improvements ✅ (Completed 2024-12-22)
- **Completed:** Fixed playground file to use proper dict access instead of attribute access
- **Features:** Updated examples/react_agent_framework_playground.py to use dictionary-style access
- **Coverage:** Fixed mypy errors and improved code quality across the codebase
- **Quality:** All linter warnings resolved, no noqa comments needed
- **Status:** Clean codebase with all tests passing and no type errors

### ReactStepStrategy Signature Fix ✅ (Completed 2024-12-22)
- **Completed:** Fixed ReactStepStrategy.react_step function signature to match LangGraph expectations
- **Features:** Renamed second parameter from "_" to "config" for proper LangGraph integration
- **Coverage:** Updated all related tests and playground examples
- **Quality:** Proper function signatures with no linter warnings
- **Status:** Full compatibility with LangGraph framework

---

## 🔄 Checkpointer Integration

* [x] **Implement basic checkpointer system** - See `CHECKPOINTER_PLAN.md` for detailed implementation plan
  - Basic integration completed (2024-12-22)
  - Remaining tasks: file system checkpointer, database checkpointer, advanced features

---

## 🎯 Current Focus Areas (2024-12-22)

### High Priority
1. **Memory Support** - Implement `.with_memory()` for state persistence
2. **Logging Integration** - Add `.with_logger()` for structured logging
3. **Conditional Execution** - Add `.with_condition()` for step guards
4. **Retry Logic** - Add `.with_retry(n)` for fault tolerance
5. **Timeout Support** - Add `.with_timeout()` for step timeouts

### Medium Priority
1. **GraphFactory** - Create multi-agent orchestration framework
2. **Tool Discovery** - Add automatic tool discovery from folders
3. **Advanced Graph Building** - Support conditional edges and branching
4. **Performance Optimization** - Profile and optimize critical paths

### Low Priority
1. **Plugin System Enhancement** - Add more plugin types and discovery
2. **Configuration Inheritance** - Support YAML configuration inheritance
3. **Advanced Features** - Encryption, compression, advanced metadata

---

End of task list.

## Discovered During Work

* [x] Fix mypy YAML stub issue by adding per-module override for `yaml` in `pyproject.toml` (2024-07-06)
    - Pre-commit and CI now pass without type: ignore comments or global ignore_missing_imports.
    - See https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports for reference.
