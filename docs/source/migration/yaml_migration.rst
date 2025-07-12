Migration Guide: Programmatic to YAML Configuration
==================================================

This guide helps you migrate from programmatic agent configuration to YAML-based configuration in the Petal framework.

Overview
--------

The Petal framework now supports loading nodes from YAML configuration files, providing a declarative approach alongside the existing programmatic API. This migration guide shows you how to convert your existing programmatic configurations to YAML format.

Benefits of YAML Configuration
-----------------------------

- **Version Control**: YAML files can be easily version controlled and shared
- **Environment Management**: Different configurations for different environments
- **Separation of Concerns**: Configuration separated from code
- **Reusability**: YAML configurations can be reused across different agents
- **Maintainability**: Easier to maintain and update configurations

Migration Steps
--------------

Step 1: Identify Your Current Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by identifying your current programmatic configuration:

.. code-block:: python

    # Current programmatic configuration
    from petal.core.factory import AgentFactory, DefaultState

    factory = AgentFactory(DefaultState)

    # LLM step
    factory.with_chat(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
        prompt_template="You are a helpful assistant. Answer: {user_input}",
        system_prompt="You are a knowledgeable and helpful AI assistant."
    )

    # React step with tools
    factory.with_react_loop(
        tools=["search", "calculator"],
        reasoning_prompt="Think step by step about how to solve this problem.",
        system_prompt="You are a reasoning agent that can use tools to solve problems.",
        max_iterations=5
    )

Step 2: Create YAML Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create separate YAML files for each node type:

**LLM Node** (``config/llm_node.yaml``):

.. code-block:: yaml

    type: llm
    name: assistant
    description: A helpful AI assistant for answering questions
    provider: openai
    model: gpt-4o-mini
    temperature: 0.0
    max_tokens: 1000
    prompt: "You are a helpful assistant. Answer: {user_input}"
    system_prompt: "You are a knowledgeable and helpful AI assistant."

**React Node** (``config/react_node.yaml``):

.. code-block:: yaml

    type: react
    name: reasoning_agent
    description: An agent that can use tools and reason step by step
    tools: [search, calculator]
    reasoning_prompt: "Think step by step about how to solve this problem."
    system_prompt: "You are a reasoning agent that can use tools to solve problems."
    max_iterations: 5

Step 3: Update Your Code
~~~~~~~~~~~~~~~~~~~~~~~~

Replace programmatic calls with YAML loading:

.. code-block:: python

    # New YAML-based configuration
    from petal.core.factory import AgentFactory, DefaultState

    factory = AgentFactory(DefaultState)

    # Load nodes from YAML
    factory.node_from_yaml("config/llm_node.yaml")
    factory.node_from_yaml("config/react_node.yaml")

    # Build agent
    agent = factory.build()

Configuration Mapping
-------------------

LLM Configuration
~~~~~~~~~~~~~~~~

**Programmatic**:

.. code-block:: python

    factory.with_chat(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
        prompt_template="Hello {name}!",
        system_prompt="You are a helpful assistant."
    )

**YAML**:

.. code-block:: yaml

    type: llm
    name: assistant
    provider: openai
    model: gpt-4o-mini
    temperature: 0.0
    max_tokens: 1000
    prompt: "Hello {name}!"
    system_prompt: "You are a helpful assistant."

React Configuration
~~~~~~~~~~~~~~~~~~

**Programmatic**:

.. code-block:: python

    factory.with_react_loop(
        tools=["search", "calculator"],
        reasoning_prompt="Think step by step.",
        system_prompt="You are a reasoning agent.",
        max_iterations=5
    )

**YAML**:

.. code-block:: yaml

    type: react
    name: reasoning_agent
    tools: [search, calculator]
    reasoning_prompt: "Think step by step."
    system_prompt: "You are a reasoning agent."
    max_iterations: 5

Custom Function Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Programmatic**:

.. code-block:: python

    async def custom_function(state: dict) -> dict:
        state["processed"] = True
        return state

    factory.add(custom_function, "custom_step")

**YAML**:

.. code-block:: yaml

    type: custom
    name: custom_step
    function_path: "my_module.custom_function"
    parameters:
      batch_size: 100
      timeout: 30

Advanced Features
----------------

MCP Tool Integration
~~~~~~~~~~~~~~~~~~~

**Programmatic**:

.. code-block:: python

    factory.with_react_loop(
        tools=["search", "mcp:filesystem", "mcp:sqlite"],
        reasoning_prompt="Analyze systematically.",
        system_prompt="You are an advanced agent.",
        max_iterations=10
    )

**YAML**:

.. code-block:: yaml

    type: react
    name: advanced_agent
    tools: [search, mcp:filesystem, mcp:sqlite]
    reasoning_prompt: "Analyze systematically."
    system_prompt: "You are an advanced agent."
    max_iterations: 10

State Schema Definition
~~~~~~~~~~~~~~~~~~~~~~

**Programmatic**:

.. code-block:: python

    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class CustomState(TypedDict):
        messages: Annotated[list, add_messages]
        user_query: str
        search_results: list
        final_answer: str

    factory = AgentFactory(CustomState)

**YAML**:

.. code-block:: yaml

    type: react
    name: advanced_agent
    state_schema:
      fields:
        user_query: str
        search_results: list
        final_answer: str
    tools: [search, calculator]
    reasoning_prompt: "Analyze systematically."
    system_prompt: "You are an advanced agent."
    max_iterations: 10

Migration Checklist
------------------

Before Migration
~~~~~~~~~~~~~~~

- [ ] Document your current programmatic configuration
- [ ] Identify all node types (LLM, React, Custom)
- [ ] List all configuration parameters
- [ ] Identify any custom functions and their dependencies
- [ ] Note any environment-specific configurations

During Migration
~~~~~~~~~~~~~~~

- [ ] Create YAML files for each node type
- [ ] Map programmatic parameters to YAML fields
- [ ] Test YAML parsing and validation
- [ ] Verify node creation from YAML
- [ ] Test agent building with YAML nodes
- [ ] Validate agent execution with YAML nodes

After Migration
~~~~~~~~~~~~~~

- [ ] Remove old programmatic configuration code
- [ ] Update documentation to reference YAML files
- [ ] Add YAML files to version control
- [ ] Create environment-specific YAML configurations
- [ ] Test in all target environments
- [ ] Update CI/CD pipelines if needed

Common Migration Patterns
------------------------

Simple LLM Agent
~~~~~~~~~~~~~~~

**Before**:

.. code-block:: python

    factory = AgentFactory(DefaultState)
    factory.with_chat(
        provider="openai",
        model="gpt-4o-mini",
        prompt_template="Hello {name}!",
        system_prompt="You are a helpful assistant."
    )
    agent = factory.build()

**After**:

.. code-block:: yaml

    # config/simple_llm.yaml
    type: llm
    name: assistant
    provider: openai
    model: gpt-4o-mini
    prompt: "Hello {name}!"
    system_prompt: "You are a helpful assistant."

.. code-block:: python

    factory = AgentFactory(DefaultState)
    factory.node_from_yaml("config/simple_llm.yaml")
    agent = factory.build()

Multi-Step Agent
~~~~~~~~~~~~~~~

**Before**:

.. code-block:: python

    factory = AgentFactory(DefaultState)

    # Custom step
    async def process_input(state: dict) -> dict:
        state["processed"] = True
        return state

    factory.add(process_input, "process")

    # LLM step
    factory.with_chat(
        provider="openai",
        model="gpt-4o-mini",
        prompt_template="Process: {processed}",
        system_prompt="You are a helpful assistant."
    )

    agent = factory.build()

**After**:

.. code-block:: yaml

    # config/process_step.yaml
    type: custom
    name: process
    function_path: "my_module.process_input"

    # config/llm_step.yaml
    type: llm
    name: assistant
    provider: openai
    model: gpt-4o-mini
    prompt: "Process: {processed}"
    system_prompt: "You are a helpful assistant."

.. code-block:: python

    factory = AgentFactory(DefaultState)
    factory.node_from_yaml("config/process_step.yaml")
    factory.node_from_yaml("config/llm_step.yaml")
    agent = factory.build()

React Agent with Tools
~~~~~~~~~~~~~~~~~~~~~

**Before**:

.. code-block:: python

    factory = AgentFactory(DefaultState)
    factory.with_react_loop(
        tools=["search", "calculator"],
        reasoning_prompt="Think step by step.",
        system_prompt="You are a reasoning agent.",
        max_iterations=5
    )
    agent = factory.build()

**After**:

.. code-block:: yaml

    # config/react_agent.yaml
    type: react
    name: reasoning_agent
    tools: [search, calculator]
    reasoning_prompt: "Think step by step."
    system_prompt: "You are a reasoning agent."
    max_iterations: 5

.. code-block:: python

    factory = AgentFactory(DefaultState)
    factory.node_from_yaml("config/react_agent.yaml")
    agent = factory.build()

Troubleshooting Migration
------------------------

Common Issues
~~~~~~~~~~~~

**Issue**: YAML file not found
**Solution**: Check file path and ensure YAML file exists

**Issue**: Validation errors
**Solution**: Verify all required fields are present in YAML

**Issue**: Import errors for custom functions
**Solution**: Ensure function path is correct and module is importable

**Issue**: Tool resolution errors
**Solution**: Verify tools are registered or MCP servers are available

**Issue**: State schema conflicts
**Solution**: Ensure state schema definitions are consistent

Testing Migration
~~~~~~~~~~~~~~~~

1. **Unit Tests**: Test YAML parsing and validation
2. **Integration Tests**: Test agent building with YAML nodes
3. **End-to-End Tests**: Test complete agent execution
4. **Performance Tests**: Ensure no performance degradation
5. **Error Handling Tests**: Test error scenarios

Best Practices
-------------

1. **Start Simple**: Begin with simple LLM nodes
2. **Incremental Migration**: Migrate one node type at a time
3. **Version Control**: Commit YAML files to version control
4. **Documentation**: Document YAML structure and options
5. **Testing**: Test thoroughly at each step
6. **Backup**: Keep programmatic configuration as backup
7. **Environment Management**: Use different YAML files for different environments

Example Migration Project
------------------------

See the ``examples/yaml/`` directory for complete migration examples:

- ``llm_node.yaml``: Basic LLM migration
- ``react_node.yaml``: React agent migration
- ``custom_node.yaml``: Custom function migration
- ``complex_node.yaml``: Advanced features migration

The examples demonstrate the complete migration process from programmatic to YAML configuration.
