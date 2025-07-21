# 🔄 Checkpointer Integration Plan for Petal Framework

## 🎯 Overview

This plan outlines the integration of LangGraph's checkpointing system into the Petal framework to enable persistent state management, resumable workflows, and enhanced debugging capabilities. The implementation follows Petal's existing architectural patterns and provides a seamless developer experience.

## ✅ Current Implementation Status

**What's Implemented (2024-12-22):**
- ✅ Simple `CheckpointerConfig` with support for memory, SQLite, and PostgreSQL
- ✅ Integration with `AgentConfig` and `AgentBuilderDirector`
- ✅ Direct use of LangGraph's built-in checkpointers (InMemorySaver, SqliteSaver, PostgresSaver)
- ✅ `with_checkpointer()` methods in both `AgentFactory` and `AgentBuilder`
- ✅ Automatic checkpointer creation and graph compilation
- ✅ Comprehensive test coverage (100% for implemented features)
- ✅ Example implementations and demos

**What's Not Yet Implemented:**
- ❌ Custom checkpointer classes (not needed - using LangGraph's)
- ❌ Thread ID generation strategies (planned enhancement)
- ❌ Enhanced Agent class with checkpoint management methods (planned enhancement)
- ❌ YAML configuration support (planned enhancement)
- ❌ Advanced features (compression, encryption, etc.) (planned enhancement)

## 📚 LangGraph Checkpointer Analysis

Based on the [LangGraph persistence documentation](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints), the key components are:

### Core Concepts
1. **Checkpointers**: Store and retrieve graph state at any point in execution
2. **Thread IDs**: Unique identifiers for conversation/workflow sessions
3. **State Persistence**: Automatic saving of state between steps
4. **Resumable Execution**: Ability to continue from any checkpoint
5. **Metadata Storage**: Additional context beyond the core state

### LangGraph Implementation
```python
# LangGraph checkpointing example
from langgraph.checkpoint.memory import MemorySaver

# Create checkpointer
checkpointer = MemorySaver()

# Use in graph compilation
graph = StateGraph(StateType)
graph.set_checkpointer(checkpointer)

# Run with thread_id for persistence
result = await graph.ainvoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "user-123"}}
)
```

## 🏗️ Petal Integration Architecture

### 1. **Checkpointer Configuration System**

#### CheckpointerConfig Model
```python
from typing import Dict, Any, Literal, Optional
from pydantic import BaseModel, Field

class CheckpointerConfig(BaseModel):
    """Simple checkpointer configuration."""

    type: Literal["memory", "postgres", "sqlite"] = Field(
        default="memory",
        description="Type of checkpointer to use"
    )
    """Type of checkpointer to use."""

    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Backend-specific configuration (e.g., database URL for postgres)"
    )
    """Backend-specific configuration (e.g., database URL for postgres)."""

    enabled: bool = Field(
        default=True,
        description="Whether checkpointing is enabled"
    )
    """Whether checkpointing is enabled."""
```

**Note**: The current implementation uses LangGraph's built-in checkpointers directly rather than custom checkpointer classes. This provides immediate compatibility with LangGraph's persistence system.

### 2. **LangGraph Checkpointer Integration**

The current implementation directly integrates with LangGraph's built-in checkpointers, providing immediate compatibility and reliability.

#### Checkpointer Creation in AgentBuilderDirector
```python
def _create_checkpointer(self, checkpointer_config: CheckpointerConfig):
    """
    Create a LangGraph checkpointer based on configuration.

    Args:
        checkpointer_config: The checkpointer configuration

    Returns:
        A LangGraph checkpointer instance

    Raises:
        ValueError: If checkpointer type is not supported
    """
    if checkpointer_config.type == "memory":
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()
    elif checkpointer_config.type == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver
        if not checkpointer_config.config:
            raise ValueError("Postgres checkpointer requires configuration (connection_string)")
        return PostgresSaver(**checkpointer_config.config)
    elif checkpointer_config.type == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver
        if not checkpointer_config.config:
            raise ValueError("SQLite checkpointer requires configuration (db_file)")
        return SqliteSaver(**checkpointer_config.config)
    else:
        raise ValueError(f"Unsupported checkpointer type: {checkpointer_config.type}")
```

**Benefits of Direct LangGraph Integration:**
- **Immediate Compatibility**: Uses LangGraph's tested and maintained checkpointers
- **No Custom Code**: Reduces maintenance burden and potential bugs
- **Feature Parity**: Access to all LangGraph checkpointer features
- **Performance**: Optimized implementations from LangGraph team

### 3. **AgentFactory Integration**

#### Enhanced AgentFactory
```python
class AgentFactory:
    """Enhanced AgentFactory with checkpointer support."""

    def with_checkpointer(self, checkpointer_config: CheckpointerConfig) -> "AgentFactory":
        """
        Configure checkpointer for state persistence.

        Args:
            checkpointer_config: The checkpointer configuration

        Returns:
            self: For method chaining
        """
        self._builder._config.set_checkpointer(checkpointer_config)
        return self
```

#### AgentBuilder Integration
```python
class AgentBuilder:
    """AgentBuilder with checkpointer support."""

    def with_checkpointer(self, checkpointer_config: CheckpointerConfig) -> "AgentBuilder":
        """
        Add checkpointer configuration to the agent.

        Args:
            checkpointer_config: The checkpointer configuration

        Returns:
            self: For method chaining

        Raises:
            ValueError: If checkpointer configuration is invalid
        """
        try:
            self._config.set_checkpointer(checkpointer_config)
        except Exception as e:
            raise ValueError(f"Invalid checkpointer configuration: {e}") from e

        return self
```

### 4. **Enhanced Agent Class**

**Note**: The current implementation does not include enhanced Agent class methods for checkpoint management. The checkpointer is integrated at the graph compilation level, and checkpoint management is handled through LangGraph's native APIs.

**Planned Enhancement**: The Agent class will be enhanced to provide convenience methods for checkpoint management, making it easier for developers to work with checkpoints without directly accessing LangGraph's APIs.

#### Planned Agent Enhancement
```python
class Agent:
    """Enhanced Agent with checkpointer support."""

    async def arun(self, state: dict, thread_id: Optional[str] = None, checkpoint_id: Optional[str] = None) -> dict:
        """Run agent with checkpointing support."""
        # Configure the run
        config = {}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id

        # Run the graph
        result = await self.graph.ainvoke(state, config=config)
        return result

    def get_state(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Any:
        """Get the current state of a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        return self.graph.get_state(config)

    def get_state_history(self, thread_id: str) -> list[Any]:
        """Get the full history of a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return list(self.graph.get_state_history(config))

    def update_state(self, thread_id: str, values: dict, checkpoint_id: Optional[str] = None, as_node: Optional[str] = None) -> None:
        """Update the state of a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        self.graph.update_state(config, values, as_node=as_node)
```

### 5. **Thread ID Generation Strategies**

**Note**: The current implementation does not include custom thread ID generation strategies. Thread IDs are managed by LangGraph's native checkpointing system and are typically provided by the user when invoking the agent.

**Planned Enhancement**: Custom thread ID generation strategies will be added to provide convenience methods for generating consistent thread IDs across different use cases.

**Important**: When using checkpointers, you **must** specify a `thread_id` as part of the `configurable` portion of the config when invoking the graph:

```python
config = {"configurable": {"thread_id": "user-123"}}
result = await graph.ainvoke(state, config=config)
```

The `thread_id` is used to:
- **Persist state**: Each thread maintains its own state history
- **Enable resumption**: You can continue conversations from where they left off
- **Support time travel**: You can replay or fork from any checkpoint in the thread
- **Provide isolation**: Different threads don't interfere with each other

#### Planned Thread ID Generators
```python
from typing import Callable, Optional
import uuid
import time

class ThreadIDGenerator:
    """Thread ID generation strategies."""

    @staticmethod
    def uuid() -> str:
        """Generate UUID-based thread ID."""
        return str(uuid.uuid4())

    @staticmethod
    def timestamp() -> str:
        """Generate timestamp-based thread ID."""
        return str(int(time.time()))

    @staticmethod
    def user_specific(user_id: str) -> str:
        """Generate user-specific thread ID."""
        return f"user-{user_id}-{int(time.time())}"

    @staticmethod
    def custom(generator_func: Callable[[], str]) -> str:
        """Use custom generator function."""
        return generator_func()

class ThreadIDManager:
    """Manage thread ID generation and validation."""

    def __init__(self, strategy: str = "uuid", **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs

    def generate(self) -> str:
        """Generate a new thread ID."""
        if self.strategy == "uuid":
            return ThreadIDGenerator.uuid()
        elif self.strategy == "timestamp":
            return ThreadIDGenerator.timestamp()
        elif self.strategy == "user_specific":
            user_id = self.kwargs.get("user_id")
            if not user_id:
                raise ValueError("user_id required for user_specific strategy")
            return ThreadIDGenerator.user_specific(user_id)
        elif self.strategy == "custom":
            generator_func = self.kwargs.get("generator_func")
            if not generator_func:
                raise ValueError("generator_func required for custom strategy")
            return ThreadIDGenerator.custom(generator_func)
        else:
            raise ValueError(f"Unknown thread ID strategy: {self.strategy}")
```

### 6. **YAML Configuration Support**

**Planned Enhancement**: YAML configuration support will be added to allow checkpointer configuration through YAML files, making it easier to configure agents declaratively.

#### Planned Checkpointer YAML Configuration
```yaml
# Example agent configuration with checkpointer
name: "conversational_agent"
state_type: "ConversationState"
checkpointer:
  type: "sqlite"
  enabled: true
  config:
    db_file: "./data/checkpoints.db"

steps:
  - type: "llm"
    name: "chat"
    config:
      provider: "openai"
      model: "gpt-4"
      prompt_template: "You are a helpful assistant. {input}"
```

#### Planned YAML Parser Enhancement
```python
class CheckpointerYAMLHandler:
    """Handle checkpointer configuration in YAML."""

    def parse_checkpointer_config(self, config: Dict[str, Any]) -> CheckpointerConfig:
        """Parse checkpointer configuration from YAML."""
        return CheckpointerConfig(**config)

    def validate_checkpointer_config(self, config: Dict[str, Any]) -> bool:
        """Validate checkpointer configuration."""
        try:
            CheckpointerConfig(**config)
            return True
        except Exception:
            return False
```

## 🚀 Implementation Plan

### ✅ Completed: Simple Checkpointer Integration (2024-12-22)
- **Completed:** Simple checkpointer configuration and integration
- **Features:**
  - `CheckpointerConfig` with support for memory, postgres, sqlite types
  - Integration with `AgentConfig` and `AgentBuilderDirector`
  - Automatic checkpointer creation and graph compilation
  - Support for enabled/disabled checkpointer configuration
- **Coverage:** 10 comprehensive tests covering all scenarios
- **Quality:** All tests passing, mypy errors are in unrelated files
- **Status:** Ready for use with LangGraph checkpointers

### Phase 1: Core Checkpointer Infrastructure (Week 1)

#### Task 1.1: Create Checkpointer Configuration ✅ (Completed 2024-12-22)
- [x] Create `src/petal/core/config/checkpointer.py` with `CheckpointerConfig`
- [x] Implement simple configuration with type, config, and enabled fields
- [x] Support for memory, postgres, sqlite types
- [x] Write comprehensive unit tests (100% coverage)

#### Task 1.2: LangGraph Integration ✅ (Completed 2024-12-22)
- [x] Integrate with LangGraph's built-in checkpointers (InMemorySaver, PostgresSaver, SqliteSaver)
- [x] Implement checkpointer creation in AgentBuilderDirector
- [x] Add proper error handling for unsupported types
- [x] Write integration tests with 100% coverage

#### Task 1.3: Thread ID Management (Planned Enhancement)
- [ ] Create `src/petal/core/checkpointer/thread_id.py` with `ThreadIDGenerator` and `ThreadIDManager`
- [ ] Implement multiple generation strategies (uuid, timestamp, user_specific, custom)
- [ ] Add validation and error handling
- [ ] Write unit tests for all strategies

### Phase 2: File System and Advanced Checkpointers (Week 2)

**Note**: The current implementation uses LangGraph's built-in checkpointers, which already provide file system and database support. Custom implementations are not needed unless specific requirements cannot be met by LangGraph's checkpointers.

#### Task 2.1: File System Checkpointer (Not Needed)
- **Status**: LangGraph's built-in checkpointers already provide file system support
- **Alternative**: Use LangGraph's SqliteSaver for file-based persistence

#### Task 2.2: Database Checkpointer (Not Needed)
- **Status**: LangGraph's PostgresSaver and SqliteSaver already provide database support
- **Alternative**: Use LangGraph's built-in database checkpointers

#### Task 2.3: Checkpointer Utilities (Planned Enhancement)
- [ ] Create `src/petal/core/checkpointer/utils.py` with helper functions
- [ ] Add checkpoint compression and decompression
- [ ] Add checkpoint validation and repair tools
- [ ] Add checkpoint migration utilities
- [ ] Write utility function tests

### Phase 3: AgentFactory Integration (Week 3)

#### Task 3.1: Enhance AgentFactory ✅ (Completed 2024-12-22)
- [x] Add `with_checkpointer()` method to `AgentFactory`
- [x] Integrate checkpointer configuration into build process
- [x] Add checkpointer validation and error handling
- [x] Maintain backward compatibility
- [x] Update existing tests
- **Status:** AgentFactory and AgentBuilder both have `with_checkpointer()` methods for consistency

#### Task 3.2: Enhance Agent Class (Planned Enhancement)
- [ ] Add checkpointer support to `Agent` class
- [ ] Implement `arun()` with proper thread_id and checkpoint_id handling
- [ ] Add state management methods (get_state, get_state_history, update_state)
- [ ] Add thread ID support in execution using LangGraph's configurable pattern
- [ ] Write integration tests for thread persistence and state management

#### Task 3.3: Update Builder Director ✅ (Completed 2024-12-22)
- [x] Modify `AgentBuilderDirector` to handle checkpointer configuration
- [x] Add checkpointer setup in graph building process
- [x] Integrate with existing step strategies
- [x] Add validation for checkpointer configuration
- [x] Update builder tests
- **Status:** All tests passing, checkpointer integration complete

### Phase 4: YAML Configuration Support (Week 4)

#### Task 4.1: YAML Parser Enhancement
- [ ] Create `src/petal/core/yaml/handlers/checkpointer.py` with `CheckpointerYAMLHandler`
- [ ] Add checkpointer configuration parsing to YAML parser
- [ ] Add validation for checkpointer YAML configuration
- [ ] Update existing YAML handlers
- [ ] Write YAML parsing tests

#### Task 4.2: Configuration Examples
- [ ] Create example YAML configurations with checkpointer
- [ ] Add checkpointer configuration to existing examples
- [ ] Create migration guide for adding checkpointer to existing configs
- [ ] Update documentation with checkpointer examples

#### Task 4.3: Integration Testing
- [ ] Create end-to-end tests with YAML configuration
- [ ] Test checkpointer persistence across agent restarts
- [ ] Test thread ID generation and management
- [ ] Test checkpoint loading and resumption
- [ ] Write comprehensive integration tests

### Phase 5: Advanced Features and Optimization (Week 5)

#### Task 5.1: Checkpoint Metadata Support
- [ ] Add metadata storage to checkpoint data
- [ ] Implement metadata filtering and querying
- [ ] Add checkpoint tagging and categorization
- [ ] Add metadata-based checkpoint search
- [ ] Write metadata handling tests

#### Task 5.2: Checkpoint Compression and Optimization
- [ ] Add automatic checkpoint compression for large states
- [ ] Implement checkpoint deduplication
- [ ] Add checkpoint cleanup and retention policies
- [ ] Add checkpoint size monitoring and alerts
- [ ] Write optimization tests

#### Task 5.3: Checkpoint Security and Privacy
- [ ] Add encryption support for sensitive checkpoint data
- [ ] Implement access control for checkpoints
- [ ] Add audit logging for checkpoint operations
- [ ] Add data retention and deletion policies
- [ ] Write security and privacy tests

### Phase 6: Documentation and Examples (Week 6)

#### Task 6.1: API Documentation
- [ ] Update API documentation with checkpointer methods
- [ ] Add checkpointer configuration examples
- [ ] Create checkpointer usage guide
- [ ] Add troubleshooting section
- [ ] Update README with checkpointer features

#### Task 6.2: Example Applications
- [ ] Create conversational agent with persistence
- [ ] Create workflow agent with checkpoint resumption
- [ ] Create multi-user agent with thread isolation
- [ ] Create long-running task agent with progress tracking
- [ ] Write example application tests

#### Task 6.3: Migration Guide
- [ ] Create migration guide from non-checkpointed to checkpointed agents
- [ ] Add performance impact analysis
- [ ] Add best practices for checkpointer usage
- [ ] Add troubleshooting common issues
- [ ] Create migration examples

## 🎯 Success Criteria

### Functional Requirements
1. **Checkpoint Persistence**: Agents can save and restore state at any point ✅
2. **Thread Isolation**: Multiple conversations/workflows are properly isolated ✅
3. **Resumable Execution**: Agents can continue from any checkpoint ✅
4. **Multiple Backends**: Support for memory, SQLite, and PostgreSQL storage ✅
5. **YAML Configuration**: Checkpointer configuration via YAML files (Future Enhancement)
6. **Backward Compatibility**: Existing agents work without modification ✅

### Performance Requirements
1. **Minimal Overhead**: Checkpointing adds <5% execution time overhead
2. **Efficient Storage**: Checkpoints are compressed and optimized
3. **Fast Retrieval**: Checkpoint loading completes in <100ms
4. **Scalable**: Support for thousands of concurrent threads

### Quality Requirements
1. **100% Test Coverage**: All checkpointer code has comprehensive tests
2. **Type Safety**: Full type hints and mypy compliance
3. **Error Handling**: Graceful handling of all error conditions
4. **Documentation**: Complete API documentation and examples

## 🔧 Technical Implementation Details

### State Serialization
```python
import pickle
import json
from typing import Any, Dict

class StateSerializer:
    """Serialize and deserialize agent state for checkpointing."""

    @staticmethod
    def serialize(state: Dict[str, Any]) -> bytes:
        """Serialize state to bytes."""
        return pickle.dumps(state)

    @staticmethod
    def deserialize(data: bytes) -> Dict[str, Any]:
        """Deserialize bytes to state."""
        return pickle.loads(data)

    @staticmethod
    def to_json(state: Dict[str, Any]) -> str:
        """Convert state to JSON string."""
        return json.dumps(state, default=str)

    @staticmethod
    def from_json(data: str) -> Dict[str, Any]:
        """Convert JSON string to state."""
        return json.loads(data)
```

### Checkpoint Data Structure
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

@dataclass
class CheckpointData:
    """Structure for checkpoint data."""

    thread_id: str
    checkpoint_id: str
    state: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "thread_id": self.thread_id,
            "checkpoint_id": self.checkpoint_id,
            "state": self.state,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointData":
        """Create from dictionary."""
        return cls(
            thread_id=data["thread_id"],
            checkpoint_id=data["checkpoint_id"],
            state=data["state"],
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data.get("version", "1.0")
        )
```

### Error Handling
```python
class CheckpointerError(Exception):
    """Base exception for checkpointer errors."""
    pass

class CheckpointNotFoundError(CheckpointerError):
    """Raised when checkpoint is not found."""
    pass

class CheckpointCorruptionError(CheckpointerError):
    """Raised when checkpoint data is corrupted."""
    pass

class CheckpointerConfigurationError(CheckpointerError):
    """Raised when checkpointer configuration is invalid."""
    pass
```

## 🚀 Usage Examples

### Thread ID and Checkpointer Usage

**Critical**: When using checkpointers, you must pass the `thread_id` in the `configurable` section of the config when invoking the graph. This is how LangGraph knows which thread to persist state to.

```python
# Correct way to use thread_id with checkpointers
config = {"configurable": {"thread_id": "user-123"}}
result = await graph.ainvoke(state, config=config)

# For resuming from a specific checkpoint
config = {"configurable": {"thread_id": "user-123", "checkpoint_id": "checkpoint-uuid"}}
result = await graph.ainvoke(state, config=config)
```

### Basic Checkpointer Usage
```python
from petal.core.factory import AgentFactory
from petal.core.config.checkpointer import CheckpointerConfig
from petal.types.state import DefaultState

# Create agent with memory checkpointer
agent = (
    AgentFactory(DefaultState)
    .with_chat(llm_config={"provider": "openai", "model": "gpt-4"})
    .with_checkpointer(CheckpointerConfig(type="memory"))
    .build()
)

# Run with thread ID for persistence (using LangGraph's native API)
thread_id = "user-123"
result = await agent.graph.ainvoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config={"configurable": {"thread_id": thread_id}}
)

# Continue the conversation in the same thread
result = await agent.graph.ainvoke(
    {"messages": [{"role": "user", "content": "What did I just say?"}]},
    config={"configurable": {"thread_id": thread_id}}
)

# Get the current state of the thread
current_state = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
print(f"Current state: {current_state.values}")

# Get the full history of the thread
history = list(agent.graph.get_state_history({"configurable": {"thread_id": thread_id}}))
print(f"Thread has {len(history)} checkpoints")
```

### SQLite Checkpointer
```python
# Create agent with SQLite checkpointer
agent = (
    AgentFactory(DefaultState)
    .with_chat(llm_config={"provider": "openai", "model": "gpt-4"})
    .with_checkpointer(CheckpointerConfig(
        type="sqlite",
        config={"db_file": "./data/checkpoints.db"}
    ))
    .build()
)
```

### Postgres Checkpointer
```python
# Create agent with Postgres checkpointer
agent = (
    AgentFactory(DefaultState)
    .with_chat(llm_config={"provider": "openai", "model": "gpt-4"})
    .with_checkpointer(CheckpointerConfig(
        type="postgres",
        config={"connection_string": "postgresql://user:pass@localhost/db"}
    ))
    .build()
)
```

### YAML Configuration (Planned Enhancement)
```yaml
# configs/agent_with_checkpointer.yaml
name: "persistent_agent"
state_type: "DefaultState"
checkpointer:
  type: "sqlite"
  enabled: true
  config:
    db_file: "./data/checkpoints.db"

steps:
  - type: "llm"
    name: "chat"
    config:
      provider: "openai"
      model: "gpt-4"
      prompt_template: "You are a helpful assistant. {input}"
```

**Note**: YAML configuration support for checkpointer is planned for future development to provide declarative configuration capabilities.

## 📊 Benefits

1. **Persistent Conversations**: Maintain conversation history across sessions
2. **Resumable Workflows**: Continue long-running tasks from any point
3. **Debugging Support**: Inspect and debug agent state at any checkpoint
4. **Multi-User Support**: Isolate conversations per user/thread
5. **Fault Tolerance**: Recover from failures by resuming from checkpoints
6. **Performance Monitoring**: Track agent performance across sessions
7. **State Analysis**: Analyze agent behavior patterns over time

## 🔄 Integration with Existing Features

### Tool Registry Integration
- Checkpoint tool usage and results
- Maintain tool state across sessions
- Resume tool execution from checkpoints

### MCP Integration
- Persist MCP server connections
- Maintain MCP tool state
- Resume MCP operations

### YAML Configuration
- Configure checkpointer via YAML
- Support checkpointer in node definitions
- Validate checkpointer configuration

### Testing Support
- Mock checkpointer for unit tests
- Test checkpoint persistence
- Test checkpoint resumption

This comprehensive plan provides a solid foundation for integrating LangGraph's checkpointing system into the Petal framework while maintaining the existing architectural patterns and developer experience.
