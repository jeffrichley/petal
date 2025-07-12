# YAML Node Configuration Guide

This directory contains example YAML configurations for creating nodes in the Petal framework.

## Overview

The Petal framework supports loading nodes from YAML configuration files, providing a declarative way to configure LLM, React, and Custom nodes alongside the existing programmatic API.

## Node Types

### LLM Nodes

LLM nodes represent language model interactions with configurable providers, models, and prompts.

**Example**: `llm_node.yaml`
```yaml
type: llm
name: assistant
description: A helpful AI assistant for answering questions
provider: openai
model: gpt-4o-mini
temperature: 0.0
max_tokens: 1000
prompt: "You are a helpful assistant. Answer the user's question: {user_input}"
system_prompt: "You are a knowledgeable and helpful AI assistant."
```

**Configuration Options**:
- `type`: Must be "llm"
- `name`: Unique identifier for the node
- `description`: Human-readable description
- `provider`: LLM provider (e.g., "openai", "anthropic")
- `model`: Model name (e.g., "gpt-4o-mini", "claude-3-sonnet")
- `temperature`: Sampling temperature (0.0 to 1.0)
- `max_tokens`: Maximum tokens to generate
- `prompt`: Template string for user prompts
- `system_prompt`: System-level instructions

### React Nodes

React nodes implement reasoning loops with tool usage capabilities.

**Example**: `react_node.yaml`
```yaml
type: react
name: reasoning_agent
description: An agent that can use tools and reason step by step
tools: [search, calculator]
reasoning_prompt: "Think step by step about how to solve this problem."
system_prompt: "You are a reasoning agent that can use tools to solve problems."
max_iterations: 5
```

**Configuration Options**:
- `type`: Must be "react"
- `name`: Unique identifier for the node
- `description`: Human-readable description
- `tools`: List of tool names or MCP references
- `reasoning_prompt`: Instructions for reasoning process
- `system_prompt`: System-level instructions
- `max_iterations`: Maximum reasoning loop iterations
- `state_schema`: Optional state schema definition

### Custom Nodes

Custom nodes allow integration of arbitrary Python functions.

**Example**: `custom_node.yaml`
```yaml
type: custom
name: data_processor
description: A custom data processing node
function_path: "examples.custom_tool.process_data"
parameters:
  batch_size: 100
  timeout: 30
validation:
  input_schema: "DataInput"
  output_schema: "DataOutput"
```

**Configuration Options**:
- `type`: Must be "custom"
- `name`: Unique identifier for the node
- `description`: Human-readable description
- `function_path`: Python import path to function
- `parameters`: Dictionary of function parameters
- `validation`: Optional input/output schema validation

### Complex Nodes

Complex nodes demonstrate advanced features like MCP tool integration and state schemas.

**Example**: `complex_node.yaml`
```yaml
type: react
name: advanced_agent
description: Advanced agent with MCP tools and state schema
tools:
  - search
  - mcp:filesystem
  - mcp:sqlite
reasoning_prompt: "Analyze the problem systematically and use available tools."
system_prompt: "You are an advanced AI agent with access to multiple tools."
max_iterations: 10
state_schema:
  fields:
    user_query:
      type: str
      description: "The user's input query"
    search_results:
      type: list
      description: "Results from search operations"
    final_answer:
      type: str
      description: "The final answer to provide"
```

## Tool References

### Direct Tool Names
```yaml
tools: [search, calculator, database]
```

### MCP Tool References
```yaml
tools: [mcp:filesystem, mcp:sqlite, mcp:github]
```

## State Schema Definition

State schemas define the structure of the agent's state:

```yaml
state_schema:
  fields:
    field_name:
      type: str|int|float|bool|list|dict
      description: "Field description"
```

## Usage

Load nodes from YAML files using the AgentFactory:

```python
from petal.core.factory import AgentFactory
from petal.types.state import DefaultState

# Create factory
factory = AgentFactory(DefaultState)

# Load node from YAML
node = factory.node_from_yaml("examples/yaml/llm_node.yaml")

# Build agent
agent = factory.build()
```

## Best Practices

1. **Use descriptive names**: Choose clear, unique names for your nodes
2. **Provide descriptions**: Always include helpful descriptions
3. **Validate configurations**: Test your YAML files before deployment
4. **Use environment variables**: For sensitive configuration like API keys
5. **Keep configurations modular**: Split complex configurations into smaller files

## Error Handling

Common YAML configuration errors:
- Missing required fields
- Invalid node types
- Malformed YAML syntax
- Invalid tool references
- Missing function paths for custom nodes

## Migration from Programmatic Configuration

To migrate from programmatic to YAML configuration:

1. Extract configuration parameters
2. Create YAML file with same structure
3. Replace programmatic calls with `node_from_yaml()`
4. Test thoroughly
5. Update documentation

## Examples

See the individual YAML files in this directory for complete working examples.
