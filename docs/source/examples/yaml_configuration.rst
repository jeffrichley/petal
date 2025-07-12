YAML Node Configuration
=======================

The Petal framework supports loading nodes from YAML configuration files, providing a declarative way to configure LLM, React, and Custom nodes alongside the existing programmatic API.

Overview
--------

YAML configuration allows you to define node configurations in a human-readable format, making it easier to:

- Share and version control configurations
- Configure nodes without writing code
- Maintain consistent node configurations across environments
- Integrate with configuration management systems

Basic Usage
-----------

Load nodes from YAML files using the AgentFactory:

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState

    # Create factory
    factory = AgentFactory(DefaultState)

    # Load node from YAML
    factory.node_from_yaml("config/llm_node.yaml")

    # Build agent
    agent = factory.build()

Node Types
----------

LLM Nodes
~~~~~~~~~

LLM nodes represent language model interactions with configurable providers, models, and prompts.

**Example**: ``llm_node.yaml``

.. code-block:: yaml

    type: llm
    name: assistant
    description: A helpful AI assistant for answering questions
    provider: openai
    model: gpt-4o-mini
    temperature: 0.0
    max_tokens: 1000
    prompt: "You are a helpful assistant. Answer the user's question: {user_input}"
    system_prompt: "You are a knowledgeable and helpful AI assistant."

**Configuration Options**:

- ``type``: Must be "llm"
- ``name``: Unique identifier for the node
- ``description``: Human-readable description
- ``provider``: LLM provider (e.g., "openai", "anthropic")
- ``model``: Model name (e.g., "gpt-4o-mini", "claude-3-sonnet")
- ``temperature``: Sampling temperature (0.0 to 1.0)
- ``max_tokens``: Maximum tokens to generate
- ``prompt``: Template string for user prompts
- ``system_prompt``: System-level instructions

React Nodes
~~~~~~~~~~~

React nodes implement reasoning loops with tool usage capabilities.

**Example**: ``react_node.yaml``

.. code-block:: yaml

    type: react
    name: reasoning_agent
    description: An agent that can use tools and reason step by step
    tools: [search, calculator]
    reasoning_prompt: "Think step by step about how to solve this problem."
    system_prompt: "You are a reasoning agent that can use tools to solve problems."
    max_iterations: 5

**Configuration Options**:

- ``type``: Must be "react"
- ``name``: Unique identifier for the node
- ``description``: Human-readable description
- ``tools``: List of tool names or MCP references
- ``reasoning_prompt``: Instructions for reasoning process
- ``system_prompt``: System-level instructions
- ``max_iterations``: Maximum reasoning loop iterations
- ``state_schema``: Optional state schema definition

Custom Nodes
~~~~~~~~~~~

Custom nodes allow integration of arbitrary Python functions.

**Example**: ``custom_node.yaml``

.. code-block:: yaml

    type: custom
    name: data_processor
    description: A custom data processing node
    function_path: "examples.custom_tool.process_data"
    parameters:
      batch_size: 100
      timeout: 30
    validation:
      input_schema:
        fields:
          data: str
          format: str
      output_schema:
        fields:
          result: str
          status: str

**Configuration Options**:

- ``type``: Must be "custom"
- ``name``: Unique identifier for the node
- ``description``: Human-readable description
- ``function_path``: Python import path to function
- ``parameters``: Dictionary of function parameters
- ``validation``: Optional input/output schema validation

Advanced Features
----------------

MCP Tool Integration
~~~~~~~~~~~~~~~~~~~

React nodes support MCP (Model Context Protocol) tool references:

.. code-block:: yaml

    type: react
    name: advanced_agent
    tools:
      - search
      - mcp:filesystem
      - mcp:sqlite
    reasoning_prompt: "Analyze the problem systematically and use available tools."
    system_prompt: "You are an advanced AI agent with access to multiple tools."
    max_iterations: 10

State Schema Definition
~~~~~~~~~~~~~~~~~~~~~~

Define dynamic state schemas for your nodes:

.. code-block:: yaml

    type: react
    name: advanced_agent
    state_schema:
      fields:
        user_query: str
        search_results: list
        final_answer: str
    tools: [search, calculator]
    reasoning_prompt: "Analyze the problem systematically."
    system_prompt: "You are an advanced AI agent."
    max_iterations: 10

Tool References
--------------

Direct Tool Names
~~~~~~~~~~~~~~~~

Reference tools by their registered names:

.. code-block:: yaml

    tools: [search, calculator, database]

MCP Tool References
~~~~~~~~~~~~~~~~~~

Reference MCP tools using the ``mcp:`` prefix:

.. code-block:: yaml

    tools: [mcp:filesystem, mcp:sqlite, mcp:github]

State Schema Definition
----------------------

State schemas define the structure of the agent's state:

.. code-block:: yaml

    state_schema:
      fields:
        field_name:
          type: str|int|float|bool|list|dict
          description: "Field description"

Supported field types:

- ``str``: String values
- ``int``: Integer values
- ``float``: Floating-point values
- ``bool``: Boolean values
- ``list``: List/array values
- ``dict``: Dictionary/object values

Best Practices
-------------

1. **Use descriptive names**: Choose clear, unique names for your nodes
2. **Provide descriptions**: Always include helpful descriptions
3. **Validate configurations**: Test your YAML files before deployment
4. **Use environment variables**: For sensitive configuration like API keys
5. **Keep configurations modular**: Split complex configurations into smaller files

Error Handling
-------------

Common YAML configuration errors:

- **Missing required fields**: Ensure all required fields are present
- **Invalid node types**: Use only supported node types (llm, react, custom)
- **Malformed YAML syntax**: Validate YAML syntax before loading
- **Invalid tool references**: Ensure tools are registered or MCP servers are available
- **Missing function paths**: Verify custom function paths exist and are importable

Example Error Messages
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    # Missing required field
    pydantic_core._pydantic_core.ValidationError: 1 validation error for LLMNodeConfig
    name
      Field required [type=missing, input_value={}, input_type=dict]

    # Invalid node type
    ValueError: Unsupported node type: unsupported

    # File not found
    petal.core.yaml.parser.YAMLFileNotFoundError: YAML file not found: config.yaml

Migration from Programmatic Configuration
---------------------------------------

To migrate from programmatic to YAML configuration:

1. **Extract configuration parameters**:

   .. code-block:: python

       # Before: Programmatic configuration
       factory.with_chat(
           provider="openai",
           model="gpt-4o-mini",
           temperature=0.0,
           prompt_template="Hello {name}!",
           system_prompt="You are a helpful assistant."
       )

2. **Create YAML file**:

   .. code-block:: yaml

       # config/llm_node.yaml
       type: llm
       name: assistant
       provider: openai
       model: gpt-4o-mini
       temperature: 0.0
       prompt: "Hello {name}!"
       system_prompt: "You are a helpful assistant."

3. **Replace programmatic calls**:

   .. code-block:: python

       # After: YAML configuration
       factory.node_from_yaml("config/llm_node.yaml")

4. **Test thoroughly**: Ensure the YAML configuration produces the same behavior

5. **Update documentation**: Document the new YAML-based approach

Configuration Examples
--------------------

See the ``examples/yaml/`` directory for complete working examples:

- ``llm_node.yaml``: Basic LLM configuration
- ``react_node.yaml``: React agent with tools
- ``custom_node.yaml``: Custom function node
- ``complex_node.yaml``: Advanced configuration with MCP tools and state schema

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Tool not found in registry
**Solution**: Ensure tools are registered or MCP servers are available

**Issue**: Invalid YAML syntax
**Solution**: Use a YAML validator to check syntax

**Issue**: Missing required fields
**Solution**: Check the configuration schema and add missing fields

**Issue**: Import errors for custom functions
**Solution**: Verify the function path and ensure the module is importable

**Issue**: Performance issues with large configurations
**Solution**: Consider splitting configurations into smaller files
