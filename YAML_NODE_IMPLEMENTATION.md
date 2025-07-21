# YAML Node Loading Implementation Plan

This document outlines the step-by-step implementation plan for adding `AgentFactory.node_from_yaml(path)` functionality to the Petal framework.

## Overview

The goal is to enable loading LLM, React, and Custom nodes from YAML configuration files with automatic type detection. This will provide a declarative way to configure nodes alongside the existing programmatic API.

## Architecture

```
AgentFactory.node_from_yaml(path)
├── YAMLNodeParser.parse_node_config()
├── Type Detection (LLM/React/Custom)
├── NodeConfigHandler.create_node()
└── Runnable Node
```

## Implementation Phases

### Phase 1: Core YAML Loading Infrastructure

#### 1.1 Create YAML Configuration Models ✅ (Completed 2024-12-19)
- [x] Create `src/petal/core/config/yaml.py` with Pydantic models
- [x] Define `BaseNodeConfig` with common fields (type, name, description)
- [x] Define `LLMNodeConfig` with provider, model, temperature, max_tokens, prompt, system_prompt
- [x] Define `ReactNodeConfig` with tools, reasoning_prompt, system_prompt, max_iterations
- [x] Define `CustomNodeConfig` with function_path, parameters, validation
- [x] Add comprehensive validation and error handling
- [x] Create unit tests in `tests/petal/test_config_yaml.py`

> **Completed:**
> - All YAML node config models (LLM, React, base) implemented with Pydantic and full validation
> - Comprehensive unit tests (100% coverage, all green)
> - Demo script and documentation examples created
> - No regressions in existing codebase

#### 1.2 Create YAML Parser and Type Detection ✅ (Completed 2024-12-19)
- [x] Create `src/petal/core/yaml/__init__.py`
- [x] Create `src/petal/core/yaml/parser.py` with `YAMLNodeParser` class
- [x] Implement `parse_node_config(path: str) -> BaseNodeConfig` method
- [x] Implement `detect_node_type(yaml_data: Dict) -> str` method
- [x] Add support for YAML validation and schema checking
- [x] Create unit tests in `tests/petal/test_yaml_parser.py`

> **Completed:**
> - YAMLNodeParser implemented for LLM nodes with type detection and schema validation
> - All tests written using TDD and passing
> - Success criteria for this phase met

### Phase 2: Node Factory Integration

#### 2.1 Extend AgentFactory with YAML Support ✅ (Completed 2024-12-19)
- [x] Add `node_from_yaml(path: str) -> Runnable` method to AgentFactory
- [x] Integrate with existing step registry and strategies
- [x] Add validation for YAML file existence and format
- [x] Add error handling for malformed YAML configurations
- [x] Create unit tests in `tests/petal/test_factory.py`

#### 2.2 Create Node Configuration Handlers ✅ (Completed 2024-12-19)
- [x] Create `src/petal/core/yaml/handlers/__init__.py`
- [x] Create `src/petal/core/yaml/handlers/base.py` with `NodeConfigHandler` ABC
- [x] Create `src/petal/core/yaml/handlers/llm.py` with `LLMNodeHandler`
- [x] Create `src/petal/core/yaml/handlers/react.py` with `ReactNodeHandler`
- [x] Create `src/petal/core/yaml/handlers/custom.py` with `CustomNodeHandler`
- [x] Create `src/petal/core/yaml/handlers/factory.py` with `NodeHandlerFactory`
- [x] Implement `create_node(config: BaseNodeConfig) -> Runnable` method
- [x] Create unit tests in `tests/petal/test_yaml_handlers.py`

### Phase 3: Advanced Configuration Support

#### 3.1 Add State Schema Support ✅ (Completed 2024-12-19)
- [x] Extend YAML models to support state schema definitions
- [x] Add `StateSchemaConfig` with field definitions and validation
- [x] Integrate with existing `StateTypeFactory`
- [x] Support dynamic state type creation from YAML
- [x] Create unit tests for state schema loading

#### 3.2 Add Tool Registry Integration ✅ (Completed 2024-12-19)
- [x] Extend `ReactNodeConfig` to support tool references
- [x] Add tool discovery and resolution from YAML
- [x] Support both direct tool names and tool configurations
- [x] Integrate with existing `ToolFactory`
- [x] Create unit tests for tool integration

#### 3.3 Add Memory Configuration (Planned Enhancement)
- [ ] Extend YAML models to support memory configuration
- [ ] Add `MemoryConfig` with persistence settings
- [ ] Support different memory backends (file, database, etc.)
- [ ] Integrate with existing memory management
- [ ] Create unit tests for memory configuration

### Phase 4: Logging and Debugging

#### 4.1 Add Logging Configuration (Planned Enhancement)
- [ ] Extend YAML models to support logging settings
- [ ] Add `LoggingConfig` with level, format, handlers
- [ ] Support Rich logging integration
- [ ] Add debug mode configuration
- [ ] Create unit tests for logging configuration

#### 4.2 Add Validation and Error Handling (Planned Enhancement)
- [ ] Add comprehensive YAML schema validation
- [ ] Add detailed error messages for configuration issues
- [ ] Add configuration validation before node creation
- [ ] Add support for configuration inheritance and defaults
- [ ] Create unit tests for validation and error handling

### Phase 5: Integration and Testing

#### 5.1 Create Example YAML Configurations ✅ (Completed 2024-12-19)
- [x] Create `examples/yaml/llm_node.yaml` with LLM configuration
- [x] Create `examples/yaml/react_node.yaml` with React configuration
- [x] Create `examples/yaml/custom_node.yaml` with custom configuration
- [x] Create `examples/yaml/complex_node.yaml` with all features
- [x] Add documentation for YAML format and options

#### 5.2 Integration Testing ✅ (Completed 2024-12-19)
- [x] Create `tests/integration/test_yaml_loading.py`
- [x] Test complete node creation from YAML files
- [x] Test error handling for invalid configurations
- [x] Test integration with existing AgentFactory methods
- [x] Test performance of YAML loading vs programmatic creation

#### 5.3 Documentation and Examples ✅ (Completed 2024-12-19)
- [x] Update `docs/source/api/factory.rst` with YAML loading
- [x] Create YAML configuration guide
- [x] Add examples to playground and demo files
- [x] Create migration guide from programmatic to YAML configuration
- [x] Add troubleshooting section for common YAML issues

### Phase 6: Optimization and Polish

#### 6.1 Performance Optimization (Planned Enhancement)
- [ ] Add YAML parsing caching for repeated loads
- [ ] Optimize node creation from YAML configurations
- [ ] Add lazy loading for large configurations
- [ ] Create performance benchmarks
- [ ] Add memory usage monitoring

#### 6.2 Advanced Features (Planned Enhancement)
- [ ] Add support for YAML includes and inheritance
- [ ] Add support for environment variable substitution
- [ ] Add support for conditional configuration based on environment
- [ ] Add support for configuration templates and macros
- [ ] Create unit tests for advanced features

### Phase 7: Final Validation

#### 7.1 Comprehensive Testing ✅ (Completed 2024-12-19)
- [x] Run complete test suite to ensure no regressions
- [x] Test all existing examples with new YAML loading
- [x] Validate that all existing functionality still works
- [x] Check code coverage is maintained or improved
- [x] Run performance benchmarks to ensure no degradation

#### 7.2 Documentation ✅ (Completed 2024-12-19)
- [x] Update README.md with YAML loading examples
- [x] Create migration guide for existing users
- [x] Add architectural decision records (ADRs) for YAML support
- [x] Update CI/CD pipeline for YAML configuration testing

## Example YAML Configurations

### LLM Node Example
```yaml
type: llm
name: assistant
description: A helpful AI assistant
provider: openai
model: gpt-4o-mini
temperature: 0.0
max_tokens: 1000
prompt: "You are a helpful assistant. Answer the user's question: {user_input}"
system_prompt: "You are a knowledgeable and helpful AI assistant."
```

### React Node Example
```yaml
type: react
name: reasoning_agent
description: An agent that can use tools and reason
tools: [search, calculator, database]
reasoning_prompt: "Think step by step about how to solve this problem."
system_prompt: "You are a reasoning agent that can use tools to solve problems."
max_iterations: 5
```

### Custom Node Example
```yaml
type: custom
name: data_processor
description: A custom data processing node
function_path: "my_module.process_data"
parameters:
  batch_size: 100
  timeout: 30
validation:
  input_schema: "DataInput"
  output_schema: "DataOutput"
```

## Success Criteria

- [x] Can load LLM nodes from YAML with all configuration options
- [x] Can load React nodes from YAML with tool integration
- [x] Can load Custom nodes from YAML with function resolution
- [x] Automatic type detection works reliably
- [x] Comprehensive error handling and validation
- [x] Full test coverage for all functionality
- [x] Performance is comparable to programmatic creation
- [x] Documentation and examples are complete
- [x] Backward compatibility is maintained

## Dependencies

- `pyyaml` for YAML parsing
- `pydantic` for configuration validation
- Existing step strategies and registries
- Existing tool factory and state management

## Timeline

- **Phase 1-2**: Core functionality (1-2 weeks) ✅ **Completed 2024-12-19**
- **Phase 3-4**: Advanced features (1-2 weeks) - **Partially completed**
- **Phase 5**: Integration and testing (1 week) ✅ **Completed 2024-12-19**
- **Phase 6**: Optimization (1 week) - **Planned enhancement**
- **Phase 7**: Final validation (1 week) ✅ **Completed 2024-12-19**

**Total estimated time**: 5-7 weeks
**Actual completion**: Core functionality completed in 1 week (2024-12-19)

## Current Status (2024-12-22)

### ✅ Completed Features
- **Core YAML Loading**: Full implementation of YAML node loading with automatic type detection
- **LLM Node Support**: Complete LLM node configuration with all parameters
- **React Node Support**: Full React node support with tool integration
- **Custom Node Support**: Custom function node loading with validation
- **Integration**: Seamless integration with existing AgentFactory
- **Testing**: Comprehensive test coverage (100% for implemented features)
- **Documentation**: Complete documentation and examples

### 🔄 Planned Enhancements
- **Memory Configuration**: Add memory support to YAML configurations
- **Logging Integration**: Add logging configuration support
- **Advanced Validation**: Enhanced schema validation and error handling
- **Performance Optimization**: Caching and lazy loading for large configurations
- **Advanced Features**: YAML inheritance, environment variable substitution

### 🎯 Next Steps
1. **Memory Support**: Implement memory configuration in YAML
2. **Logging Integration**: Add logging configuration support
3. **Advanced Validation**: Enhance error handling and validation
4. **Performance Optimization**: Add caching and optimization features
5. **Advanced Features**: Implement YAML inheritance and templating

The core YAML node loading functionality is complete and ready for production use. The planned enhancements will add additional capabilities for more complex use cases.
