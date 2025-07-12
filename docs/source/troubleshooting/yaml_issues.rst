Troubleshooting YAML Configuration Issues
========================================

This guide helps you diagnose and resolve common issues when working with YAML node configurations in the Petal framework.

Common Error Messages
--------------------

YAML File Not Found
~~~~~~~~~~~~~~~~~~~

**Error**:
.. code-block:: text

    petal.core.yaml.parser.YAMLFileNotFoundError: YAML file not found: config.yaml

**Cause**: The specified YAML file doesn't exist at the given path.

**Solution**:
1. Check the file path is correct
2. Ensure the file exists in the specified location
3. Use absolute paths if relative paths are causing issues
4. Verify file permissions allow reading the file

**Example**:
.. code-block:: python

    # Wrong path
    factory.node_from_yaml("config.yaml")

    # Correct path
    factory.node_from_yaml("examples/yaml/llm_node.yaml")

Invalid YAML Syntax
~~~~~~~~~~~~~~~~~~

**Error**:
.. code-block:: text

    petal.core.yaml.parser.YAMLParseError: Invalid YAML syntax in config.yaml: expected '<document start>'

**Cause**: The YAML file contains syntax errors.

**Solution**:
1. Validate YAML syntax using a YAML validator
2. Check for missing quotes around strings
3. Verify indentation is consistent (use spaces, not tabs)
4. Ensure proper YAML structure

**Example**:
.. code-block:: yaml

    # Invalid YAML
    type: llm
    name: assistant
    provider: openai
    model: gpt-4o-mini
    temperature: 0.0
    max_tokens: 1000
    prompt: "Hello {name}!"
    system_prompt: "You are a helpful assistant."

    # Valid YAML
    type: llm
    name: assistant
    provider: openai
    model: gpt-4o-mini
    temperature: 0.0
    max_tokens: 1000
    prompt: "Hello {name}!"
    system_prompt: "You are a helpful assistant."

Missing Required Fields
~~~~~~~~~~~~~~~~~~~~~~

**Error**:
.. code-block:: text

    pydantic_core._pydantic_core.ValidationError: 1 validation error for LLMNodeConfig
    name
      Field required [type=missing, input_value={}, input_type=dict]

**Cause**: Required fields are missing from the YAML configuration.

**Solution**:
1. Check the configuration schema for required fields
2. Add missing required fields to the YAML file
3. Ensure field names match exactly (case-sensitive)

**Required Fields by Node Type**:

LLM Nodes:
- `type`: Must be "llm"
- `name`: Unique identifier
- `provider`: LLM provider (e.g., "openai")
- `model`: Model name

React Nodes:
- `type`: Must be "react"
- `name`: Unique identifier
- `tools`: List of tool names

Custom Nodes:
- `type`: Must be "custom"
- `name`: Unique identifier
- `function_path`: Python import path

Invalid Node Type
~~~~~~~~~~~~~~~~

**Error**:
.. code-block:: text

    ValueError: Unsupported node type: unsupported

**Cause**: The node type specified is not supported.

**Solution**:
1. Use only supported node types: "llm", "react", "custom"
2. Check for typos in the type field
3. Ensure the type field is a string

**Example**:
.. code-block:: yaml

    # Invalid
    type: unsupported
    name: test

    # Valid
    type: llm
    name: assistant

Invalid Provider
~~~~~~~~~~~~~~~

**Error**:
.. code-block:: text

    pydantic_core._pydantic_core.ValidationError: 1 validation error for LLMNodeConfig
    provider
      String should match pattern '^(openai|anthropic|google|cohere|huggingface)$'

**Cause**: The provider specified is not supported.

**Solution**:
1. Use only supported providers: "openai", "anthropic", "google", "cohere", "huggingface"
2. Check for typos in the provider name
3. Ensure the provider is correctly configured

Invalid Temperature
~~~~~~~~~~~~~~~~~~

**Error**:
.. code-block:: text

    pydantic_core._pydantic_core.ValidationError: 1 validation error for LLMNodeConfig
    temperature
      Value error, temperature must be between 0.0 and 2.0

**Cause**: Temperature value is outside the valid range.

**Solution**:
1. Use temperature values between 0.0 and 2.0
2. Ensure temperature is a number, not a string

**Example**:
.. code-block:: yaml

    # Invalid
    temperature: 3.0

    # Valid
    temperature: 0.7

Tool Not Found
~~~~~~~~~~~~~~

**Error**:
.. code-block:: text

    KeyError: "Tool 'search' not found in registry."

**Cause**: The specified tool is not registered in the tool factory.

**Solution**:
1. Ensure the tool is registered before using it
2. Check tool name spelling
3. For MCP tools, ensure the MCP server is available
4. Register tools using the ToolFactory

**Example**:
.. code-block:: python

    from petal.core.tool_factory import ToolFactory

    # Register tools
    tool_factory = ToolFactory()
    tool_factory.register("search", search_function)
    tool_factory.register("calculator", calculator_function)

Import Error for Custom Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error**:
.. code-block:: text

    ImportError: Failed to import function 'my_module.custom_function': No module named 'my_module'

**Cause**: The custom function cannot be imported.

**Solution**:
1. Verify the function path is correct
2. Ensure the module is in the Python path
3. Check that the function exists in the module
4. Test the import manually

**Example**:
.. code-block:: yaml

    # Invalid path
    function_path: "my_module.custom_function"

    # Valid path (if module exists)
    function_path: "examples.custom_tool.process_data"

State Schema Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error**:
.. code-block:: text

    pydantic_core._pydantic_core.ValidationError: 1 validation error for ReactNodeConfig
    state_schema.fields.user_query
      Input should be a valid string

**Cause**: State schema fields are incorrectly formatted.

**Solution**:
1. Use string type names for state schema fields
2. Ensure field names are valid Python identifiers
3. Use supported field types: str, int, float, bool, list, dict

**Example**:
.. code-block:: yaml

    # Invalid
    state_schema:
      fields:
        user_query:
          type: str
          description: "User query"

    # Valid
    state_schema:
      fields:
        user_query: str
        search_results: list
        final_answer: str

Performance Issues
-----------------

Slow YAML Loading
~~~~~~~~~~~~~~~~~

**Symptoms**: YAML files take a long time to load or parse.

**Solutions**:
1. Keep YAML files small and focused
2. Use caching for frequently loaded configurations
3. Consider lazy loading for large configurations
4. Profile YAML parsing performance

Large Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Memory usage is high when loading large YAML files.

**Solutions**:
1. Split large configurations into smaller files
2. Use includes or references to share common configurations
3. Implement lazy loading for large configurations
4. Consider using a database for very large configurations

Debugging Techniques
-------------------

Enable Debug Logging
~~~~~~~~~~~~~~~~~~~

Add debug logging to see what's happening during YAML loading:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.DEBUG)

    from petal.core.factory import AgentFactory, DefaultState

    factory = AgentFactory(DefaultState)
    factory.node_from_yaml("config.yaml")

Validate YAML Files
~~~~~~~~~~~~~~~~~~

Use a YAML validator to check syntax:

.. code-block:: python

    import yaml

    def validate_yaml(file_path):
        try:
            with open(file_path, 'r') as f:
                yaml.safe_load(f)
            print(f"✓ {file_path} is valid YAML")
        except yaml.YAMLError as e:
            print(f"✗ {file_path} has YAML errors: {e}")

Test Configuration Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~

Test YAML parsing without building the agent:

.. code-block:: python

    from petal.core.yaml.parser import YAMLNodeParser

    parser = YAMLNodeParser()
    try:
        config = parser.parse_node_config("config.yaml")
        print(f"✓ Configuration parsed successfully: {config}")
    except Exception as e:
        print(f"✗ Configuration parsing failed: {e}")

Common Patterns
--------------

Environment-Specific Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use different YAML files for different environments:

.. code-block:: python

    import os

    env = os.getenv("ENVIRONMENT", "development")
    config_file = f"config/{env}/llm_node.yaml"

    factory = AgentFactory(DefaultState)
    factory.node_from_yaml(config_file)

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~

Validate configurations before using them:

.. code-block:: python

    from petal.core.config.yaml import LLMNodeConfig, ReactNodeConfig, CustomNodeConfig

    def validate_config(config_data, config_type):
        try:
            if config_type == "llm":
                LLMNodeConfig(**config_data)
            elif config_type == "react":
                ReactNodeConfig(**config_data)
            elif config_type == "custom":
                CustomNodeConfig(**config_data)
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

Error Recovery
~~~~~~~~~~~~~

Implement error recovery for YAML loading:

.. code-block:: python

    def load_node_safely(factory, yaml_path, fallback_config=None):
        try:
            return factory.node_from_yaml(yaml_path)
        except FileNotFoundError:
            print(f"YAML file not found: {yaml_path}")
            if fallback_config:
                return factory.with_chat(**fallback_config)
        except Exception as e:
            print(f"Error loading YAML: {e}")
            if fallback_config:
                return factory.with_chat(**fallback_config)
        return None

Best Practices
-------------

1. **Use Descriptive Names**: Choose clear, unique names for your nodes
2. **Provide Descriptions**: Always include helpful descriptions
3. **Validate Early**: Test YAML files before deployment
4. **Use Environment Variables**: For sensitive configuration like API keys
5. **Keep Configurations Modular**: Split complex configurations into smaller files
6. **Version Control**: Commit YAML files to version control
7. **Documentation**: Document YAML structure and options
8. **Testing**: Test thoroughly at each step

Prevention Strategies
--------------------

1. **Schema Validation**: Use Pydantic models to validate configurations
2. **Automated Testing**: Write tests for YAML configurations
3. **CI/CD Integration**: Validate YAML files in CI/CD pipelines
4. **Code Review**: Review YAML changes as part of code review
5. **Documentation**: Keep documentation up to date with YAML changes

Getting Help
-----------

If you're still experiencing issues:

1. **Check the Documentation**: Review the YAML configuration guide
2. **Search Issues**: Look for similar issues in the project repository
3. **Create Minimal Example**: Create a minimal YAML file that reproduces the issue
4. **Provide Error Details**: Include full error messages and stack traces
5. **Share Configuration**: Share your YAML configuration (without sensitive data)

Example Debugging Session
------------------------

.. code-block:: python

    # 1. Enable debug logging
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # 2. Validate YAML syntax
    import yaml
    with open("config.yaml", "r") as f:
        data = yaml.safe_load(f)
        print("YAML syntax is valid")

    # 3. Test configuration parsing
    from petal.core.yaml.parser import YAMLNodeParser
    parser = YAMLNodeParser()
    config = parser.parse_node_config("config.yaml")
    print(f"Configuration parsed: {config}")

    # 4. Test node creation
    from petal.core.factory import AgentFactory, DefaultState
    factory = AgentFactory(DefaultState)
    node = factory.node_from_yaml("config.yaml")
    print("Node created successfully")

    # 5. Test agent building
    agent = factory.build()
    print("Agent built successfully")
