# 🏗️ Factory.py Architectural Refactoring - Micro-Tasks

This document breaks down the factory.py architectural refactoring into small, LLM-manageable tasks that are finishable and testable.

## Overview

The refactoring transforms the monolithic `AgentFactory` into a composable, extensible system using design patterns:
- **Strategy Pattern** for step management
- **Configuration Object Pattern** for configuration management
- **Builder Pattern** with composition
- **Chain of Responsibility** for step configuration
- **Registry Pattern** for extensibility

## Testing Strategy & Breaking Changes

### Expected Breaking Changes:
1. **Import Changes**: Some classes will move to new modules
2. **Method Signatures**: Some builder methods may have slightly different signatures
3. **Internal Structure**: The way steps are managed internally will change

### Mitigation Strategy:
1. **Backward Compatibility Layer**: Maintain old interface while using new architecture internally
2. **Gradual Migration**: Keep old methods working during transition with deprecation warnings
3. **Comprehensive Test Updates**: Update tests as we go, not all at once

### Risk Mitigation:
- Each micro-task includes test updates
- Integration tests verify end-to-end functionality
- Rollback plan with old code in separate branch
- Optional feature flags for gradual rollout

---

## Phase 1A: Foundation Setup

### Task 1.1: Create Step Strategy Base (1 hour)
**Goal**: Create the foundation for pluggable step strategies

**Files to create/modify**:
- [x] `src/petal/core/steps/__init__.py`
- [x] `src/petal/core/steps/base.py`
- [x] `tests/petal/test_steps_base.py`

**Sample Code**:
```python
# src/petal/core/steps/base.py
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

class StepStrategy(ABC):
    """Abstract base class for step creation strategies."""

    @abstractmethod
    def create_step(self, config: Dict[str, Any]) -> Callable:
        """Create a step callable from configuration."""
        pass

    @abstractmethod
    def get_node_name(self, index: int) -> str:
        """Generate a node name for the step at the given index."""
        pass

# Usage example:
class MyCustomStrategy(StepStrategy):
    def create_step(self, config: Dict[str, Any]) -> Callable:
        return config["step_function"]

    def get_node_name(self, index: int) -> str:
        return f"custom_step_{index}"

# Test that it works:
strategy = MyCustomStrategy()
step = strategy.create_step({"step_function": lambda x: x})
node_name = strategy.get_node_name(0)
assert node_name == "custom_step_0"
```

**Deliverables**:
- [x] `StepStrategy` ABC with abstract methods:
  - [x] `create_step(config: Dict[str, Any]) -> Callable`
  - [x] `get_node_name(index: int) -> str`
- [x] Comprehensive type hints and docstrings
- [x] Unit tests with 100% coverage
- [x] All tests passing

**Success Criteria**:
- [x] ABC cannot be instantiated directly
- [x] Concrete implementations work correctly
- [x] All abstract methods are enforced
- [x] Type hints are complete and accurate
- [x] Docstrings follow Google style
- [x] Tests cover all code paths

**Test Requirements**:
```python
def test_step_strategy_abc():
    # Test that ABC cannot be instantiated
    # Test that abstract methods are enforced
    # Test that concrete implementations work
```

### Task 1.2: Create Step Registry (1 hour)
**Goal**: Create a registry system for step type discovery

**Files to create/modify**:
- [x] `src/petal/core/steps/registry.py`
- [x] `tests/petal/test_steps_registry.py`

**Sample Code**:
```python
# src/petal/core/steps/registry.py
from typing import Dict, Type
from .base import StepStrategy

class StepRegistry:
    """Registry for step creation strategies."""

    def __init__(self):
        self._strategies: Dict[str, Type[StepStrategy]] = {}
        self._register_defaults()

    def register(self, name: str, strategy: Type[StepStrategy]) -> None:
        """Register a step strategy."""
        self._strategies[name] = strategy

    def get_strategy(self, name: str) -> StepStrategy:
        """Get a step strategy by name."""
        if name not in self._strategies:
            raise ValueError(f"Unknown step type: {name}")
        return self._strategies[name]()

    def _register_defaults(self) -> None:
        """Register built-in strategies."""
        # Will be implemented in later tasks
        pass

# Usage example:
registry = StepRegistry()
registry.register("custom", MyCustomStrategy)
strategy = registry.get_strategy("custom")
step = strategy.create_step({"step_function": lambda x: x})
```

**Deliverables**:
- [x] `StepRegistry` class with methods:
  - [x] `register(name: str, strategy: Type[StepStrategy])`
  - [x] `get_strategy(name: str) -> StepStrategy`
  - [x] `_register_defaults()` for built-in strategies
- [x] Validation and error handling for unknown step types
- [x] Unit tests with 100% coverage
- [x] All tests passing

**Success Criteria**:
- [x] Can register new strategies
- [x] Can retrieve registered strategies
- [x] Throws appropriate error for unknown strategies
- [x] Default strategies are registered automatically
- [x] Thread-safe registration and retrieval
- [x] All error cases are handled

**Test Requirements**:
```python
def test_step_registry():
    # Test registration of strategies
    # Test retrieval of strategies
    # Test error handling for unknown types
    # Test default registration
```

### Task 1.3: Extract LLMStep to Strategy (2 hours)
**Goal**: Move LLMStep to new architecture and create LLMStepStrategy

**Files to create/modify**:
- [x] `src/petal/core/steps/llm.py`
- [x] `src/petal/core/factory.py` (remove LLMStep class)
- [x] `tests/petal/test_steps_llm.py`
- [x] Update imports in existing files

**Sample Code**:
```python
# src/petal/core/steps/llm.py
from typing import Any, Callable, Dict, Optional
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from .base import StepStrategy

class LLMStep:
    """Encapsulates the configuration and logic for an LLM step."""

    def __init__(self, prompt_template: str, system_prompt: str,
                 llm_config: Dict[str, Any], llm_instance: Optional[BaseChatModel]):
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.llm_config = llm_config
        self.llm_instance = llm_instance

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Existing LLMStep logic here
        pass

class LLMStepStrategy(StepStrategy):
    """Strategy for creating LLM steps."""

    def create_step(self, config: Dict[str, Any]) -> Callable:
        """Create an LLMStep from configuration."""
        prompt_template = config.get("prompt_template", "")
        system_prompt = config.get("system_prompt", "")
        llm_config = config.get("llm_config", {})
        llm_instance = config.get("llm_instance")

        return LLMStep(prompt_template, system_prompt, llm_config, llm_instance)

    def get_node_name(self, index: int) -> str:
        """Generate node name for LLM step."""
        return f"llm_step_{index}"

# Usage example:
strategy = LLMStepStrategy()
llm_step = strategy.create_step({
    "prompt_template": "Hello {name}",
    "system_prompt": "You are a helpful assistant",
    "llm_config": {"model": "gpt-4o-mini"}
})
node_name = strategy.get_node_name(0)  # "llm_step_0"
```
**Note:** TDD was used for this task. Coverage for new code is high. All tests for new and refactored code pass.

**Deliverables**:
- [x] Move `LLMStep` class from `factory.py` to `steps/llm.py`
- [x] Create `LLMStepStrategy` implementing `StepStrategy`
- [x] Update all imports and fix broken references
- [x] Comprehensive validation for LLM configuration
- [x] Unit tests with 100% coverage for new code
- [x] All existing tests still passing

**Success Criteria**:
- [x] LLMStep class is successfully moved
- [x] LLMStepStrategy implements StepStrategy correctly
- [x] All imports are updated and working
- [x] LLM configuration validation works
- [x] All existing functionality preserved
- [x] No breaking changes to public API

**Test Requirements**:
```python
def test_llm_step_strategy():
    # Test LLMStep creation from config
    # Test node name generation
    # Test LLM configuration validation
    # Test all LLM provider configurations
```

---

## Phase 1B: Configuration Objects

### Task 1.4: Create AgentConfig (1 hour)
**Goal**: Separate configuration from building logic

**Files to create/modify**:
- `src/petal/core/config/__init__.py`
- `src/petal/core/config/agent.py`
- `tests/petal/test_config_agent.py`

**Sample Code**:
```python
# src/petal/core/config/agent.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from ..steps.base import StepStrategy

@dataclass
class AgentConfig:
    """Configuration object for agent building."""

    state_type: type
    steps: List[Tuple[StepStrategy, Dict[str, Any]]] = field(default_factory=list)
    memory: Optional[Dict[str, Any]] = None
    graph_config: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, strategy: StepStrategy, config: Dict[str, Any]) -> None:
        """Add a step to the configuration."""
        self.steps.append((strategy, config))

    def set_memory(self, memory_config: Dict[str, Any]) -> None:
        """Set memory configuration."""
        self.memory = memory_config

    def validate(self) -> None:
        """Validate configuration integrity."""
        if not self.steps:
            raise ValueError("Agent must have at least one step")
        # Add more validation as needed

# Usage example:
config = AgentConfig(state_type=DefaultState)
config.add_step(LLMStepStrategy(), {
    "prompt_template": "Hello {name}",
    "llm_config": {"model": "gpt-4o-mini"}
})
config.set_memory({"type": "conversation"})
config.validate()
```

**Deliverables**:
- [ ] `AgentConfig` dataclass with fields:
  - `state_type: type`
  - `steps: List[Tuple[StepStrategy, Dict[str, Any]]]`
  - `memory: Optional[Dict[str, Any]]`
  - `graph_config: Dict[str, Any]`
- [ ] Methods: `add_step()`, `set_memory()`, `validate()`
- [ ] Validation methods for configuration integrity
- [ ] Unit tests with 100% coverage
- [ ] All tests passing

**Success Criteria**:
- [ ] Dataclass can be created with required fields
- [ ] Steps can be added and retrieved
- [ ] Memory configuration can be set
- [ ] Validation catches configuration errors
- [ ] All fields have proper type hints
- [ ] Immutable where appropriate

**Test Requirements**:
```python
def test_agent_config():
    # Test dataclass creation
    # Test add_step method
    # Test set_memory method
    # Test validation methods
```

### Task 1.5: Create StateTypeFactory (1.5 hours)
**Goal**: Centralize state type creation logic

**Files to create/modify**:
- `src/petal/core/config/state.py`
- `src/petal/core/factory.py` (move logic)
- `tests/petal/test_config_state.py`

**Sample Code**:
```python
# src/petal/core/config/state.py
from typing import Dict, type
from typing_extensions import get_type_hints
from langgraph.graph.message import MessagesState

class StateTypeFactory:
    """Factory for creating state types with message support."""

    _cache: Dict[tuple, type] = {}

    @classmethod
    def create_with_messages(cls, base_type: type) -> type:
        """Create a state type that includes message support."""
        type_hints = get_type_hints(base_type, include_extras=True)

        if "messages" in type_hints:
            return base_type

        # Create cached key
        cache_key = (base_type.__name__, tuple(sorted(type_hints.items())))

        if cache_key in cls._cache:
            return cls._cache[cache_key]

        # Create combined type
        dynamic_name = f"{base_type.__name__}WithMessagesAddedByPetal"
        combined_type = type(dynamic_name, (base_type, MessagesState), {})

        cls._cache[cache_key] = combined_type
        return combined_type

    @classmethod
    def create_mergeable(cls, base_type: type) -> type:
        """Create a state type that supports merging."""
        # Implementation for mergeable state types
        return base_type

# Usage example:
from typing import TypedDict

class MyState(TypedDict):
    name: str

# Create state with messages
state_with_messages = StateTypeFactory.create_with_messages(MyState)
# Now state_with_messages includes message support
```

**Deliverables**:
- [ ] `StateTypeFactory` class with static methods:
  - `create_with_messages(base_type: type) -> type`
  - `create_mergeable(base_type: type) -> type`
- [ ] Move `_create_state_type()` logic from `AgentFactory`
- [ ] Caching mechanism for dynamic type creation
- [ ] Comprehensive error handling for type creation failures
- [ ] Unit tests with 100% coverage
- [ ] All tests passing

**Success Criteria**:
- [ ] Can create state types with message support
- [ ] Caching works correctly and improves performance
- [ ] Error handling for invalid state types
- [ ] No memory leaks from caching
- [ ] All existing state type logic preserved
- [ ] Type hints are accurate and complete

**Test Requirements**:
```python
def test_state_type_factory():
    # Test create_with_messages
    # Test create_mergeable
    # Test caching mechanism
    # Test error handling
```

---

## Phase 1C: Builder Foundation

### Task 1.6: Create AgentBuilder (1.5 hours)
**Goal**: Create fluent interface using composition

**Files to create/modify**:
- `src/petal/core/builders/__init__.py`
- `src/petal/core/builders/agent.py`
- `tests/petal/test_builders_agent.py`

**Sample Code**:
```python
# src/petal/core/builders/agent.py
from typing import Any, Dict
from ..config.agent import AgentConfig
from ..steps.registry import StepRegistry

class AgentBuilder:
    """Fluent interface for building agents."""

    def __init__(self, state_type: type):
        self._config = AgentConfig(state_type)
        self._registry = StepRegistry()

    def with_step(self, step_type: str, **config: Any) -> "AgentBuilder":
        """Add a step to the agent."""
        strategy = self._registry.get_strategy(step_type)
        self._config.add_step(strategy, config)
        return self

    def with_memory(self, memory_config: Dict[str, Any]) -> "AgentBuilder":
        """Add memory configuration to the agent."""
        self._config.set_memory(memory_config)
        return self

    def build(self) -> "Agent":
        """Build the agent from configuration."""
        from ..factory import AgentFactory
        # Use AgentFactory to build from config
        # This will be implemented in later tasks
        pass

# Usage example:
builder = AgentBuilder(DefaultState)
agent = (builder
    .with_step("llm", prompt_template="Hello {name}", model="gpt-4o-mini")
    .with_memory({"type": "conversation"})
    .build())
```

**Deliverables**:
- [ ] `AgentBuilder` class with fluent interface:
  - `with_step(step_type: str, **config) -> AgentBuilder`
  - `with_memory(memory_config: Dict[str, Any]) -> AgentBuilder`
- [ ] Use composition with `AgentConfig` and `StepRegistry`
- [ ] Validation for builder state consistency
- [ ] Unit tests with 100% coverage
- [ ] All tests passing

**Success Criteria**:
- [ ] Fluent interface works correctly
- [ ] Steps can be configured with various options
- [ ] Memory configuration works
- [ ] Validation catches invalid configurations
- [ ] Method chaining works properly
- [ ] All configuration is properly stored

**Test Requirements**:
```python
def test_agent_builder():
    # Test fluent interface
    # Test step configuration
    # Test memory configuration
    # Test validation
```

### Task 1.7: Create AgentBuilderDirector (2 hours)
**Goal**: Move complex building logic to dedicated director

**Files to create/modify**:
- `src/petal/core/builders/director.py`
- `src/petal/core/factory.py` (move logic)
- `tests/petal/test_builders_director.py`

**Sample Code**:
```python
# src/petal/core/builders/director.py
from typing import Any
from langgraph.graph import END, START, StateGraph
from ..config.agent import AgentConfig
from ..config.state import StateTypeFactory
from ..factory import Agent

class AgentBuilderDirector:
    """Director for building agents from configuration."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def build(self) -> Agent:
        """Build an agent from configuration."""
        # Create state type
        state_type = self._create_state_type()

        # Build graph
        graph = self._build_graph(state_type)

        # Create agent
        return Agent().build(graph, state_type)

    def _create_state_type(self) -> type:
        """Create the appropriate state type."""
        if self._has_chat_model():
            return StateTypeFactory.create_with_messages(self.config.state_type)
        return self.config.state_type

    def _build_graph(self, state_type: type) -> StateGraph:
        """Build the LangGraph StateGraph."""
        graph = StateGraph(state_type)

        # Add nodes
        for i, (strategy, config) in enumerate(self.config.steps):
            step = strategy.create_step(config)
            node_name = strategy.get_node_name(i)
            graph.add_node(node_name, step)

            # Add edges
            if i == 0:
                graph.add_edge(START, node_name)
            else:
                prev_node = self.config.steps[i-1][1].get_node_name(i-1)
                graph.add_edge(prev_node, node_name)

            if i == len(self.config.steps) - 1:
                graph.add_edge(node_name, END)

        return graph.compile()

    def _has_chat_model(self) -> bool:
        """Check if any steps are LLM steps."""
        return any("llm" in str(type(strategy)) for strategy, _ in self.config.steps)

# Usage example:
config = AgentConfig(DefaultState)
config.add_step(LLMStepStrategy(), {"prompt_template": "Hello"})
director = AgentBuilderDirector(config)
agent = director.build()
```

**Deliverables**:
- [ ] `AgentBuilderDirector` class with methods:
  - `build() -> Agent`
  - `_create_state_type()`
  - `_build_graph()`
- [ ] Move complex building logic from `AgentFactory.build()`
- [ ] Comprehensive error handling for build failures
- [ ] Unit tests with 100% coverage
- [ ] All tests passing

**Success Criteria**:
- [ ] Can build agents from configuration
- [ ] State type creation works correctly
- [ ] Graph building creates proper LangGraph
- [ ] Error handling for build failures
- [ ] All existing building logic preserved
- [ ] Performance is maintained or improved

**Test Requirements**:
```python
def test_agent_builder_director():
    # Test build method
    # Test state type creation
    # Test graph building
    # Test error handling
```

---

## Phase 2A: Integration

### Task 2.1: Update AgentFactory to Use New Architecture (2 hours)
**Goal**: Integrate new architecture while maintaining backward compatibility

**Files to modify**:
- `src/petal/core/factory.py`
- `tests/petal/test_factory.py` (update existing tests)

**Sample Code**:
```python
# Updated AgentFactory using new architecture
class AgentFactory:
    """Builder and fluent interface for constructing Agent objects."""

    def __init__(self, state_type: type):
        self._builder = AgentBuilder(state_type)

    def add(self, step: Callable[..., Any], node_name: Optional[str] = None) -> "AgentFactory":
        """Add an async step to the agent."""
        # Use new architecture internally
        self._builder.with_step("custom", step=step, node_name=node_name)
        return self

    def with_chat(self, llm: Optional[Union[BaseChatModel, Dict[str, Any]]] = None, **kwargs) -> "ChatStepBuilder":
        """Add an LLM step to the chain."""
        # Maintain backward compatibility
        config = {"llm": llm, **kwargs}
        self._builder.with_step("llm", **config)
        return ChatStepBuilder(self, len(self._builder._config.steps) - 1)

    def with_memory(self, memory: Optional[Any] = None) -> "AgentFactory":
        """Add memory support to the agent."""
        self._builder.with_memory({"memory": memory})
        return self

    def build(self) -> Agent:
        """Build the agent."""
        director = AgentBuilderDirector(self._builder._config)
        return director.build()

# Usage example (backward compatible):
factory = AgentFactory(DefaultState)
agent = (factory
    .with_chat()
    .with_prompt("Hello {name}")
    .add(lambda x: x)
    .build())
```

**Deliverables**:
- [ ] Modify `AgentFactory` to use `AgentBuilder` composition
- [ ] Update `with_chat()` method to use new step registry
- [ ] Update `add()` method to use new step strategies
- [ ] Maintain backward compatibility during transition
- [ ] All existing tests still passing
- [ ] Unit tests with 100% coverage

**Success Criteria**:
- [ ] All existing public API methods work unchanged
- [ ] New architecture is used internally
- [ ] No breaking changes to existing code
- [ ] Performance is maintained
- [ ] All existing tests pass
- [ ] New functionality works correctly

**Test Requirements**:
```python
def test_agent_factory_backward_compatibility():
    # Test existing with_chat() method still works
    # Test existing add() method still works
    # Test all existing functionality preserved
```

### Task 2.2: Add CustomStepStrategy (1 hour)
**Goal**: Support arbitrary callable functions as steps

**Files to create/modify**:
- `src/petal/core/steps/custom.py`
- `tests/petal/test_steps_custom.py`

**Sample Code**:
```python
# src/petal/core/steps/custom.py
from typing import Any, Callable, Dict
from .base import StepStrategy

class CustomStepStrategy(StepStrategy):
    """Strategy for creating custom function steps."""

    def create_step(self, config: Dict[str, Any]) -> Callable:
        """Create a custom step from configuration."""
        step_func = config.get("step")
        if not callable(step_func):
            raise ValueError("Custom step must be callable")
        return step_func

    def get_node_name(self, index: int) -> str:
        """Generate node name for custom step."""
        return f"custom_step_{index}"

# Usage example:
def my_custom_function(state: Dict[str, Any]) -> Dict[str, Any]:
    state["processed"] = True
    return state

strategy = CustomStepStrategy()
step = strategy.create_step({"step": my_custom_function})
node_name = strategy.get_node_name(0)  # "custom_step_0"

# Can be used in builder:
builder = AgentBuilder(DefaultState)
builder.with_step("custom", step=my_custom_function)
```

**Deliverables**:
- [ ] `CustomStepStrategy` class implementing `StepStrategy`
- [ ] Support arbitrary callable functions as steps
- [ ] Validation for step function signatures
- [ ] Support both sync and async functions
- [ ] Unit tests with 100% coverage
- [ ] All tests passing

**Success Criteria**:
- [ ] Can create steps from arbitrary callables
- [ ] Validates function signatures
- [ ] Supports both sync and async functions
- [ ] Generates appropriate node names
- [ ] Error handling for invalid functions
- [ ] Works with existing builder pattern

**Test Requirements**:
```python
def test_custom_step_strategy():
    # Test sync function support
    # Test async function support
    # Test signature validation
    # Test node name generation
```

---

## Phase 2B: Advanced Features

### Task 2.3: Add Configuration Handlers (1.5 hours)
**Goal**: Implement Chain of Responsibility for step configuration

**Files to create/modify**:
- `src/petal/core/builders/handlers/__init__.py`
- `src/petal/core/builders/handlers/base.py`
- `src/petal/core/builders/handlers/llm.py`
- `src/petal/core/builders/handlers/custom.py`
- `tests/petal/test_builders_handlers.py`

**Sample Code**:
```python
# src/petal/core/builders/handlers/base.py
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

class StepConfigHandler(ABC):
    """Abstract base class for step configuration handlers."""

    def __init__(self, next_handler: Optional["StepConfigHandler"] = None):
        self.next_handler = next_handler

    @abstractmethod
    def can_handle(self, step_type: str) -> bool:
        """Check if this handler can handle the step type."""
        pass

    @abstractmethod
    def handle(self, config: Dict[str, Any]) -> Callable:
        """Handle step configuration."""
        pass

    def process(self, step_type: str, config: Dict[str, Any]) -> Callable:
        """Process configuration through the chain."""
        if self.can_handle(step_type):
            return self.handle(config)
        elif self.next_handler:
            return self.next_handler.process(step_type, config)
        else:
            raise ValueError(f"No handler found for step type: {step_type}")

# src/petal/core/builders/handlers/llm.py
class LLMConfigHandler(StepConfigHandler):
    """Handler for LLM step configuration."""

    def can_handle(self, step_type: str) -> bool:
        return step_type == "llm"

    def handle(self, config: Dict[str, Any]) -> Callable:
        from ..steps.llm import LLMStepStrategy
        strategy = LLMStepStrategy()
        return strategy.create_step(config)

# Usage example:
llm_handler = LLMConfigHandler()
custom_handler = CustomConfigHandler(llm_handler)

# Process through chain
step = custom_handler.process("llm", {"prompt_template": "Hello"})
```

**Deliverables**:
- [ ] `StepConfigHandler` ABC with Chain of Responsibility pattern
- [ ] `LLMConfigHandler` for LLM step configuration
- [ ] `CustomConfigHandler` for custom step configuration
- [ ] Comprehensive error handling and validation
- [ ] Unit tests with 100% coverage
- [ ] All tests passing

**Success Criteria**:
- [ ] Chain of responsibility works correctly
- [ ] Each handler can process its step type
- [ ] Error handling for unknown step types
- [ ] Handlers can be chained together
- [ ] Configuration validation works
- [ ] Performance is acceptable

**Test Requirements**:
```python
def test_config_handlers():
    # Test chain of responsibility
    # Test LLM handler
    # Test custom handler
    # Test error handling
```

### Task 2.4: Add Plugin System (2 hours)
**Goal**: Create extensible plugin system for step types

**Files to create/modify**:
- `src/petal/core/plugins/__init__.py`
- `src/petal/core/plugins/base.py`
- `tests/petal/test_plugins.py`

**Sample Code**:
```python
# src/petal/core/plugins/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from ..steps.base import StepStrategy

class StepPlugin(ABC):
    """Base class for step type plugins."""

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this step type."""
        pass

    @abstractmethod
    def get_strategy(self) -> Type[StepStrategy]:
        """Get the strategy class for this step type."""
        pass

    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """Get the configuration schema for this step type."""
        pass

class PluginManager:
    """Manager for step type plugins."""

    def __init__(self):
        self._plugins: Dict[str, StepPlugin] = {}

    def register(self, plugin: StepPlugin) -> None:
        """Register a plugin."""
        self._plugins[plugin.get_name()] = plugin

    def discover(self, package_name: str) -> None:
        """Discover plugins in a package."""
        # Implementation for automatic discovery
        pass

    def get_plugin(self, name: str) -> StepPlugin:
        """Get a plugin by name."""
        if name not in self._plugins:
            raise ValueError(f"Plugin not found: {name}")
        return self._plugins[name]

# Example plugin:
class MyCustomPlugin(StepPlugin):
    def get_name(self) -> str:
        return "my_custom"

    def get_strategy(self) -> Type[StepStrategy]:
        return MyCustomStrategy

    def get_config_schema(self) -> Dict[str, Any]:
        return {"step": "callable", "config": "dict"}

# Usage:
manager = PluginManager()
manager.register(MyCustomPlugin())
plugin = manager.get_plugin("my_custom")
strategy = plugin.get_strategy()
```

**Deliverables**:
- [ ] Plugin interface and discovery system
- [ ] Automatic discovery of step type plugins
- [ ] Plugin registration and management system
- [ ] Example plugins for common step types
- [ ] Unit tests with 100% coverage
- [ ] All tests passing

**Success Criteria**:
- [ ] Plugins can be registered and retrieved
- [ ] Automatic discovery works
- [ ] Plugin management is thread-safe
- [ ] Example plugins work correctly
- [ ] Error handling for missing plugins
- [ ] Performance is acceptable

**Test Requirements**:
```python
def test_plugin_system():
    # Test plugin discovery
    # Test plugin registration
    # Test plugin management
    # Test example plugins
```

---

## Phase 3: Cleanup and Optimization

### Task 3.1: Remove ChatStepBuilder (1 hour)
**Goal**: Clean up deprecated code

**Files to modify**:
- `src/petal/core/factory.py` (remove ChatStepBuilder)
- Update all examples and documentation
- Update tests

**Sample Code**:
```python
# Before (deprecated):
factory = AgentFactory(DefaultState)
factory.with_chat().with_prompt("Hello").build()

# After (new pattern):
factory = AgentFactory(DefaultState)
factory.with_chat(prompt_template="Hello").build()

# Or using the new builder:
builder = AgentBuilder(DefaultState)
builder.with_step("llm", prompt_template="Hello")
agent = builder.build()
```

**Deliverables**:
- [ ] Remove `ChatStepBuilder` class completely
- [ ] Update all examples to use new builder pattern
- [ ] Update documentation
- [ ] Update tests to reflect new architecture
- [ ] All tests passing

**Success Criteria**:
- [ ] ChatStepBuilder is completely removed
- [ ] All examples use new patterns
- [ ] Documentation is updated
- [ ] No references to deprecated code
- [ ] All functionality preserved through new patterns
- [ ] Code is cleaner and more maintainable

**Test Requirements**:
```python
def test_no_chat_step_builder():
    # Verify ChatStepBuilder is removed
    # Test that new patterns work correctly
    # Test that examples still work
```

### Task 3.2: Performance Optimization (1.5 hours)
**Goal**: Optimize performance and add benchmarks

**Files to modify**:
- Various performance-critical files
- Add performance benchmarks

**Sample Code**:
```python
# Performance benchmarks
import time
from typing import Dict, Any

def benchmark_agent_creation():
    """Benchmark agent creation performance."""
    start_time = time.time()

    # Create agent with new architecture
    builder = AgentBuilder(DefaultState)
    for i in range(10):
        builder.with_step("llm", prompt_template=f"Step {i}")
    agent = builder.build()

    end_time = time.time()
    return end_time - start_time

def benchmark_step_registry():
    """Benchmark step registry performance."""
    registry = StepRegistry()

    # Test registration performance
    start_time = time.time()
    for i in range(1000):
        registry.register(f"step_{i}", CustomStepStrategy)
    registration_time = time.time() - start_time

    # Test retrieval performance
    start_time = time.time()
    for i in range(1000):
        registry.get_strategy(f"step_{i}")
    retrieval_time = time.time() - start_time

    return registration_time, retrieval_time

# Usage in tests:
def test_performance():
    creation_time = benchmark_agent_creation()
    assert creation_time < 1.0  # Should complete within 1 second

    reg_time, ret_time = benchmark_step_registry()
    assert reg_time < 0.1  # Registration should be fast
    assert ret_time < 0.1  # Retrieval should be fast
```

**Deliverables**:
- [ ] Profile new architecture for performance bottlenecks
- [ ] Optimize step creation and configuration
- [ ] Optimize state type creation and caching
- [ ] Add performance benchmarks and monitoring
- [ ] Create performance regression tests
- [ ] All tests passing with no performance regression

**Success Criteria**:
- [ ] Performance is maintained or improved
- [ ] No memory leaks
- [ ] Startup time is acceptable
- [ ] Runtime performance is good
- [ ] Benchmarks are comprehensive
- [ ] Performance regression tests pass

**Test Requirements**:
```python
def test_performance():
    # Benchmark old vs new architecture
    # Test no performance regression
    # Test memory usage
    # Test startup time
```

---

## Execution Order

### Recommended Sequence:
1. **Phase 1A** (Foundation) - Independent, low-risk tasks
2. **Phase 1B** (Configuration) - Builds on foundation
3. **Phase 1C** (Builder) - Brings it all together
4. **Phase 2A** (Integration) - Most critical, requires careful testing
5. **Phase 2B** (Advanced) - Optional enhancements
6. **Phase 3** (Cleanup) - Final polish

### Success Criteria for Each Task:
- [ ] All new code has comprehensive tests
- [ ] All existing tests still pass
- [ ] No performance regression
- [ ] Code follows project style guidelines
- [ ] Documentation is updated
- [ ] Examples work correctly

### Rollback Plan:
- Each task can be reverted independently
- Keep old code in separate branch
- Feature flags available for gradual rollout
- Comprehensive test coverage ensures issues are caught early

---

## Notes

- Each micro-task is designed to be **finishable** in 1-2 hours
- Each task includes **specific test requirements**
- Tasks are **independent** where possible
- **Backward compatibility** is maintained throughout
- **Performance** is monitored at each step
- **Documentation** is updated incrementally

This approach minimizes risk while maximizing progress and maintainability.
