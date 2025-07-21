# Current Tasking Status (2024-12-22)

## 🎯 Overview

The Petal framework has undergone a comprehensive architectural refactoring and now has a solid foundation with:
- ✅ Complete AgentFactory with new architecture
- ✅ Full YAML node loading support
- ✅ Basic checkpointer integration
- ✅ Comprehensive tool factory with MCP support
- ✅ 753+ unit tests with 100% coverage for implemented features

## 🏗️ Completed Major Features

### AgentFactory Core Implementation ✅
- [x] Create `AgentFactory` class
- [x] Implement `.add(step)` for function/step registration
- [x] Add `.with_prompt()` for f-string prompt injection
- [x] Add `.with_system_prompt()` to define system role
- [x] Add `.with_chat()` to configure LLM support
- [x] Add `.with_state()` for schema enforcement
- [x] Add `.with_tool_registry()` to inject tools
- [x] Add `.with_mcp_proxy()` for deferred MCP toolkit resolution
- [x] Add `.build()` to compile final agent Runnable
- [x] Add `.run()` to execute with state dict
- [x] Add `.as_node()` for LangGraph compatibility
- [x] Add `AgentFactory.node_from_yaml(path)` to load nodes from YAML

### ToolFactory Core Implementation ✅
- [x] Create `ToolFactory` class
- [x] Implement `.add(name, fn)` for direct registration
- [x] Implement `.resolve(name)` to retrieve tool
- [x] Implement `.list()` to show all registered tools
- [x] Implement `.add_mcp(name, resolver)` for mcp related tools
- [x] Support `mcp:` style namespacing

### Architectural Refactoring ✅
- [x] Strategy Pattern for step management
- [x] Configuration Object Pattern for configuration management
- [x] Builder Pattern with composition
- [x] Chain of Responsibility for step configuration
- [x] Registry Pattern for extensibility
- [x] Plugin system for step types

### YAML Node Loading ✅
- [x] Core YAML loading infrastructure
- [x] LLM, React, and Custom node support
- [x] Automatic type detection
- [x] Tool registry integration
- [x] Comprehensive validation and error handling

### Checkpointer Integration ✅
- [x] Basic checkpointer configuration
- [x] Integration with AgentFactory and AgentBuilder
- [x] Support for memory, SQLite, and PostgreSQL
- [x] Direct use of LangGraph's built-in checkpointers

## 🎯 Current Focus Areas

### High Priority - AgentFactory Pluggable Behaviors

Complete these in roughly the order they're listed—each builds on the previous:

#### 1. `.with_memory()` - Memory Support
**Status**: Not yet implemented
**Goal**: Add a strategy for persisting or recalling past state (e.g., an in-memory buffer or external vector store).

**Implementation Plan**:
- [ ] Create `MemoryConfig` with persistence settings
- [ ] Support different memory backends (file, database, etc.)
- [ ] Integrate with existing memory management
- [ ] Add `.with_memory()` method to AgentFactory
- [ ] Create unit tests for memory configuration

#### 2. `.with_logger()` - Internal State Logging
**Status**: Not yet implemented
**Goal**: Hook up structured logging of inputs, outputs, and intermediate state for debugging or audit trails.

**Implementation Plan**:
- [ ] Create `LoggingConfig` with level, format, handlers
- [ ] Support Rich logging integration
- [ ] Add debug mode configuration
- [ ] Add `.with_logger()` method to AgentFactory
- [ ] Create unit tests for logging configuration

#### 3. `.with_condition()` - Conditional Step Execution
**Status**: Not yet implemented
**Goal**: Allow users to register a predicate so a step only runs when condition(state) is truthy.

**Implementation Plan**:
- [ ] Create `ConditionConfig` with predicate functions
- [ ] Add conditional routing logic to step execution
- [ ] Support multiple condition types (state-based, time-based, etc.)
- [ ] Add `.with_condition()` method to AgentFactory
- [ ] Create unit tests for conditional execution

#### 4. `.with_retry(n)` - Retry Logic
**Status**: Not yet implemented
**Goal**: Wrap a step in retry logic (exponential back-off or fixed delay).

**Implementation Plan**:
- [ ] Create `RetryConfig` with retry strategies
- [ ] Implement exponential back-off and fixed delay
- [ ] Add retry wrapper to step execution
- [ ] Add `.with_retry(n)` method to AgentFactory
- [ ] Create unit tests for retry logic

#### 5. `.with_timeout()` - Timeout Support
**Status**: Not yet implemented
**Goal**: Enforce per-step timeouts—e.g., cancel or skip if a step takes too long.

**Implementation Plan**:
- [ ] Create `TimeoutConfig` with timeout settings
- [ ] Implement timeout wrapper for step execution
- [ ] Support different timeout behaviors (cancel, skip, retry)
- [ ] Add `.with_timeout()` method to AgentFactory
- [ ] Create unit tests for timeout handling

### Medium Priority - Framework Extensions

#### GraphFactory - Multi-Agent Orchestration
**Status**: Not yet implemented
**Goal**: Create GraphFactory for wiring multi-agent flows using `.as_node()`.

**Implementation Plan**:
- [ ] Create `GraphFactory` class
- [ ] Add `.add_node(name, node)` to wire LangGraph nodes
- [ ] Add `.add_agent(name, agent_factory)` with `.as_node()` call
- [ ] Add `.connect(src, dest)` to define DAG edges
- [ ] Implement `.build()` to compile LangGraph
- [ ] Add `.run(initial_state)`

#### Tool Discovery - Automatic Tool Loading
**Status**: Not yet implemented
**Goal**: Add automatic tool discovery from folders and files.

**Implementation Plan**:
- [ ] Add `.discover(from_folders)` for directory discovery
- [ ] Add `.discover(from_files)` for file-level loading
- [ ] Add `.from_folder(path)` shortcut
- [ ] Auto-register all `@tool_fn` decorated functions on import
- [ ] Enable agent discovery through annotated agent types

### Low Priority - Advanced Features

#### Advanced Graph Building
**Status**: Not yet implemented
**Goal**: Support conditional edges, branching logic, and parallel execution.

**Implementation Plan**:
- [ ] Create `src/petal/core/graph/__init__.py`
- [ ] Create `src/petal/core/graph/builder.py` with advanced graph building
- [ ] Support conditional edges and branching logic
- [ ] Support parallel execution paths
- [ ] Add graph visualization capabilities

#### Performance Optimization
**Status**: Not yet implemented
**Goal**: Profile and optimize critical paths for better performance.

**Implementation Plan**:
- [ ] Profile new architecture for performance bottlenecks
- [ ] Optimize step creation and configuration
- [ ] Optimize state type creation and caching
- [ ] Add performance benchmarks and monitoring
- [ ] Create performance regression tests

## 🚀 Why Focus on AgentFactory First?

It's the heart of your orchestration.

Having a fully-featured AgentFactory makes writing integration tests and example graphs much easier.

ToolFactory and GraphFactory both depend on `.with_chat()`, `.with_tool_registry()`, and `.as_node()` to integrate cleanly.

## 📊 Current Metrics

- **Test Coverage**: 753+ unit tests with 100% coverage for implemented features
- **Code Quality**: All mypy, ruff, and black checks pass
- **Documentation**: Complete API documentation and examples
- **Backward Compatibility**: All existing functionality preserved
- **Performance**: No regression in agent creation or execution

## 🎯 Next Steps

1. **Start with Memory Support** - Implement `.with_memory()` for state persistence
2. **Add Logging Integration** - Implement `.with_logger()` for structured logging
3. **Implement Conditional Execution** - Add `.with_condition()` for step guards
4. **Add Retry Logic** - Implement `.with_retry(n)` for fault tolerance
5. **Add Timeout Support** - Implement `.with_timeout()` for step timeouts

Once these core AgentFactory features are complete, we can move on to GraphFactory and advanced tool discovery features.

Let me know which of these you'd like to tackle first, or if you want a deep dive on any specific implementation!
