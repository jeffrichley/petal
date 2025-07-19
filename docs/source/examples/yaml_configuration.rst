YAML Configuration
==================

Petal supports declarative agent configuration using YAML files, providing a clean separation between configuration and code.

Overview
--------

YAML configuration allows you to define agents declaratively with comprehensive validation and type safety. The system supports multiple node types, state schemas, and tool discovery configuration.

Key Features
------------

- **Multiple Node Types**: LLM, ReAct, and Custom nodes
- **State Schema Definition**: Dynamic state type creation
- **Tool Discovery**: Configurable tool discovery strategies
- **Validation**: Comprehensive Pydantic-based validation
- **Environment Variables**: Support for environment variable substitution
- **Template Variables**: Dynamic value substitution
- **Error Handling**: Clear error messages for invalid configurations

Node Types
----------

LLM Node
~~~~~~~~

Basic LLM node configuration:

.. code-block:: yaml

    type: llm
    name: assistant
    description: A helpful assistant
    provider: openai
    model: gpt-4o-mini
    temperature: 0.7
    max_tokens: 1000
    prompt: "Help with {task}"
    system_prompt: "You are a helpful assistant."

Advanced LLM configuration:

.. code-block:: yaml

    type: llm
    name: creative_writer
    description: A creative writing assistant
    provider: openai
    model: gpt-4o-mini
    temperature: 0.9
    max_tokens: 2000
    prompt: "Write a creative story about {topic}"
    system_prompt: |
      You are a creative writer with a vivid imagination.
      Write engaging, descriptive stories that captivate readers.
    enabled: true
    state_schema:
      fields:
        topic: str
        genre: str
        word_count: int
      required_fields: [topic]
      optional_fields:
        genre: "fantasy"
        word_count: 500

ReAct Node
~~~~~~~~~~

ReAct reasoning node with tools:

.. code-block:: yaml

    type: react
    name: research_assistant
    description: A research assistant that can use tools
    tools: ["web:search", "calculator:add", "api:get_stock_price"]
    reasoning_prompt: |
      Think step by step about how to answer the question:
      1. Understand what information is needed
      2. Use appropriate tools to gather data
      3. Perform calculations if needed
      4. Provide a clear, comprehensive answer
    system_prompt: "You are an expert researcher with access to various tools."
    max_iterations: 8
    enabled: true

Advanced ReAct configuration:

.. code-block:: yaml

    type: react
    name: financial_advisor
    description: A financial advisor with market analysis capabilities
    tools: ["api:get_stock_price", "api:get_market_data", "calculator:calculate"]
    reasoning_prompt: |
      Analyze the financial situation step by step:
      1. Gather current market data
      2. Calculate relevant metrics
      3. Consider risk factors
      4. Provide actionable advice
    system_prompt: |
      You are a professional financial advisor.
      Provide sound, well-reasoned financial advice.
    max_iterations: 10
    mcp_servers:
      filesystem:
        command: "mcp-server-filesystem"
      database:
        command: "mcp-server-database"
        args: ["--host", "localhost", "--port", "5432"]
    state_schema:
      fields:
        investment_amount: float
        risk_tolerance: str
        time_horizon: str
        portfolio: dict
      required_fields: [investment_amount, risk_tolerance]
      optional_fields:
        time_horizon: "medium"
        portfolio: {}

Custom Node
~~~~~~~~~~~

Custom node with function import:

.. code-block:: yaml

    type: custom
    name: data_processor
    description: A custom data processing node
    function_path: "my_module.process_data"
    parameters:
      threshold: 0.5
      normalize: true
      output_format: "json"
    enabled: true

Advanced custom node:

.. code-block:: yaml

    type: custom
    name: ml_predictor
    description: A machine learning prediction node
    function_path: "ml_models.predict"
    parameters:
      model_path: "models/classifier.pkl"
      confidence_threshold: 0.8
      batch_size: 32
    validation:
      input_schema:
        fields:
          features: list
          metadata: dict
        required_fields: [features]
      output_schema:
        fields:
          predictions: list
          confidence: list
          model_version: str
        required_fields: [predictions, confidence]
      state_schema:
        fields:
          input_data: dict
          predictions: dict
          processing_time: float
        required_fields: [input_data]
        optional_fields:
          processing_time: 0.0

State Schema Configuration
-------------------------

Dynamic state schema creation with validation:

.. code-block:: yaml

    state_schema:
      fields:
        user_input: str
        processed_data: dict
        results: list
        metadata: dict
        timestamp: str
      required_fields: [user_input]
      optional_fields:
        processed_data: {}
        results: []
        metadata: {}
        timestamp: ""
      nested_schemas:
        processed_data:
          fields:
            text: str
            tokens: list
            embeddings: list
          required_fields: [text]
          optional_fields:
            tokens: []
            embeddings: []

Complex nested schemas:

.. code-block:: yaml

    state_schema:
      fields:
        request: dict
        response: dict
        context: dict
        history: list
      required_fields: [request]
      optional_fields:
        response: {}
        context: {}
        history: []
      nested_schemas:
        request:
          fields:
            query: str
            parameters: dict
            user_id: str
          required_fields: [query, user_id]
          optional_fields:
            parameters: {}
        response:
          fields:
            content: str
            confidence: float
            sources: list
            metadata: dict
          required_fields: [content]
          optional_fields:
            confidence: 1.0
            sources: []
            metadata: {}

Tool Discovery Configuration
---------------------------

Configure tool discovery for YAML nodes:

.. code-block:: yaml

    type: llm
    name: tool_assistant
    provider: openai
    model: gpt-4o-mini
    prompt: "Use available tools to help with {task}"
    system_prompt: "You are a helpful assistant with access to tools."
    tool_discovery:
      enabled: true
      folders: ["tools/", "my_tools/"]
      config_locations: ["config/tools.yaml"]
      exclude_patterns: ["*_test.py", "*.pyc"]

Using YAML Configuration
-----------------------

Loading YAML Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Load a node from a YAML file:

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState

    # Load node from YAML
    agent = (
        AgentFactory(DefaultState)
        .node_from_yaml("config/llm_node.yaml")
        .build()
    )

    # Run the agent
    result = await agent.arun({
        "name": "User",
        "task": "Write a Python function",
        "messages": []
    })

Multiple YAML Files
~~~~~~~~~~~~~~~~~~~

Load multiple nodes from different YAML files:

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState

    agent = (
        AgentFactory(DefaultState)
        .node_from_yaml("config/llm_node.yaml")
        .node_from_yaml("config/react_node.yaml")
        .build()
    )

Error Handling
~~~~~~~~~~~~~~

Handle YAML configuration errors:

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState
    from petal.core.yaml.parser import YAMLFileNotFoundError, YAMLParseError

    try:
        agent = (
            AgentFactory(DefaultState)
            .node_from_yaml("config/agent.yaml")
            .build()
        )
    except YAMLFileNotFoundError as e:
        print(f"YAML file not found: {e}")
    except YAMLParseError as e:
        print(f"Invalid YAML syntax: {e}")
    except ValueError as e:
        print(f"Configuration error: {e}")

Validation Examples
------------------

Field Validation
~~~~~~~~~~~~~~~~

Validate required and optional fields:

.. code-block:: yaml

    type: llm
    name: validator
    provider: openai
    model: gpt-4o-mini
    state_schema:
      fields:
        required_field: str
        optional_field: int
        list_field: list
        dict_field: dict
      required_fields: [required_field]
      optional_fields:
        optional_field: 0
        list_field: []
        dict_field: {}

Type Validation
~~~~~~~~~~~~~~~

Validate field types:

.. code-block:: yaml

    type: llm
    name: type_validator
    provider: openai
    model: gpt-4o-mini
    state_schema:
      fields:
        string_field: str
        int_field: int
        float_field: float
        bool_field: bool
        list_field: list
        dict_field: dict
      required_fields: [string_field, int_field]
      optional_fields:
        float_field: 0.0
        bool_field: false
        list_field: []
        dict_field: {}

Provider Validation
~~~~~~~~~~~~~~~~~~

Validate LLM provider configuration:

.. code-block:: yaml

    type: llm
    name: provider_test
    provider: openai  # Valid: openai, anthropic, google, cohere, huggingface
    model: gpt-4o-mini
    temperature: 0.7  # Valid: 0.0 to 2.0
    max_tokens: 1000  # Valid: positive integer

Temperature Validation
~~~~~~~~~~~~~~~~~~~~~

Validate temperature range:

.. code-block:: yaml

    type: llm
    name: temperature_test
    provider: openai
    model: gpt-4o-mini
    temperature: 0.5  # Valid: 0.0 to 2.0
    # temperature: 2.5  # Invalid: exceeds maximum

Max Tokens Validation
~~~~~~~~~~~~~~~~~~~~~

Validate max_tokens:

.. code-block:: yaml

    type: llm
    name: tokens_test
    provider: openai
    model: gpt-4o-mini
    max_tokens: 1000  # Valid: positive integer
    # max_tokens: 0    # Invalid: must be positive

Max Iterations Validation
~~~~~~~~~~~~~~~~~~~~~~~~~

Validate max_iterations for ReAct nodes:

.. code-block:: yaml

    type: react
    name: iterations_test
    tools: ["calculator:add"]
    max_iterations: 5  # Valid: positive integer
    # max_iterations: 0  # Invalid: must be positive

Function Path Validation
~~~~~~~~~~~~~~~~~~~~~~~~

Validate function paths for custom nodes:

.. code-block:: yaml

    type: custom
    name: function_test
    function_path: "my_module.my_function"  # Valid: dot notation
    # function_path: ""  # Invalid: cannot be empty

Advanced Configuration
---------------------

Environment Variable Substitution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use environment variables in configuration:

.. code-block:: yaml

    type: llm
    name: env_test
    provider: openai
    model: gpt-4o-mini
    api_key: ${OPENAI_API_KEY}
    base_url: ${OPENAI_BASE_URL}

Template Variable Substitution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use template variables for dynamic values:

.. code-block:: yaml

    type: llm
    name: template_test
    provider: openai
    model: gpt-4o-mini
    prompt: "Help with {task} using {context}"
    system_prompt: "You are a {role} assistant."

Conditional Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Use conditional configuration based on environment:

.. code-block:: yaml

    type: llm
    name: conditional_test
    provider: ${LLM_PROVIDER:-openai}
    model: ${LLM_MODEL:-gpt-4o-mini}
    temperature: ${LLM_TEMPERATURE:-0.7}
    enabled: ${ENABLE_AGENT:-true}

Complex State Schemas
~~~~~~~~~~~~~~~~~~~~

Define complex state schemas with nested structures:

.. code-block:: yaml

    type: llm
    name: complex_state
    provider: openai
    model: gpt-4o-mini
    state_schema:
      fields:
        user: dict
        session: dict
        context: dict
        history: list
      required_fields: [user]
      optional_fields:
        session: {}
        context: {}
        history: []
      nested_schemas:
        user:
          fields:
            id: str
            name: str
            preferences: dict
          required_fields: [id, name]
          optional_fields:
            preferences: {}
        session:
          fields:
            session_id: str
            start_time: str
            data: dict
          required_fields: [session_id]
          optional_fields:
            start_time: ""
            data: {}

Integration Examples
-------------------

MCP Integration
~~~~~~~~~~~~~~~

Configure MCP servers in YAML:

.. code-block:: yaml

    type: react
    name: mcp_assistant
    tools: ["mcp:filesystem:read_file", "mcp:database:query"]
    reasoning_prompt: "Use MCP tools to help with the task."
    system_prompt: "You are an assistant with MCP tool access."
    mcp_servers:
      filesystem:
        command: "mcp-server-filesystem"
        args: ["--config", "fs_config.json"]
      database:
        command: "mcp-server-database"
        args: ["--host", "localhost", "--port", "5432"]

Tool Discovery Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure tool discovery in YAML:

.. code-block:: yaml

    type: llm
    name: discovery_assistant
    provider: openai
    model: gpt-4o-mini
    prompt: "Use available tools to help with {task}"
    system_prompt: "You are a helpful assistant with tool access."
    tool_discovery:
      enabled: true
      folders: ["tools/", "my_tools/"]
      config_locations: ["config/tools.yaml", "config/custom_tools.yaml"]
      exclude_patterns: ["*_test.py", "*.pyc", "temp_*"]

Multi-Node Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Configure multiple nodes in a single YAML file:

.. code-block:: yaml

    # multi_node.yaml
    nodes:
      - type: llm
        name: analyzer
        provider: openai
        model: gpt-4o-mini
        prompt: "Analyze {input}"
        system_prompt: "You are an analyzer."

      - type: react
        name: processor
        tools: ["calculator:add", "api:fetch_data"]
        reasoning_prompt: "Process the analysis step by step."
        system_prompt: "You are a processor."

      - type: custom
        name: formatter
        function_path: "utils.format_output"
        parameters:
          format: "json"
          indent: 2

Best Practices
--------------

Configuration Organization
~~~~~~~~~~~~~~~~~~~~~~~~~

Organize your YAML configurations:

.. code-block:: yaml

    # config/agents/llm_assistant.yaml
    type: llm
    name: llm_assistant
    provider: openai
    model: gpt-4o-mini
    prompt: "Help with {task}"
    system_prompt: "You are a helpful assistant."

    # config/agents/react_assistant.yaml
    type: react
    name: react_assistant
    tools: ["calculator:add", "web:search"]
    reasoning_prompt: "Think step by step."
    system_prompt: "You are a reasoning assistant."

    # config/agents/custom_processor.yaml
    type: custom
    name: custom_processor
    function_path: "processors.data_processor"
    parameters:
      threshold: 0.5

Validation Strategy
~~~~~~~~~~~~~~~~~~

Implement comprehensive validation:

.. code-block:: yaml

    type: llm
    name: validated_assistant
    provider: openai
    model: gpt-4o-mini
    temperature: 0.7
    max_tokens: 1000
    prompt: "Help with {task}"
    system_prompt: "You are a helpful assistant."
    state_schema:
      fields:
        task: str
        context: dict
        user_id: str
      required_fields: [task, user_id]
      optional_fields:
        context: {}
    validation:
      input_schema:
        fields:
          task: str
          user_id: str
        required_fields: [task, user_id]
      output_schema:
        fields:
          response: str
          confidence: float
        required_fields: [response]
        optional_fields:
          confidence: 1.0

Error Handling Strategy
~~~~~~~~~~~~~~~~~~~~~~

Implement proper error handling:

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState
    from petal.core.yaml.parser import YAMLFileNotFoundError, YAMLParseError
    import logging

    logger = logging.getLogger(__name__)

    def load_agent_from_yaml(yaml_path: str):
        try:
            agent = (
                AgentFactory(DefaultState)
                .node_from_yaml(yaml_path)
                .build()
            )
            return agent
        except YAMLFileNotFoundError as e:
            logger.error(f"YAML file not found: {yaml_path}")
            raise
        except YAMLParseError as e:
            logger.error(f"Invalid YAML syntax in {yaml_path}: {e}")
            raise
        except ValueError as e:
            logger.error(f"Configuration error in {yaml_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {yaml_path}: {e}")
            raise

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

Optimize YAML configuration for performance:

.. code-block:: yaml

    type: llm
    name: optimized_assistant
    provider: openai
    model: gpt-4o-mini
    temperature: 0.0  # Lower temperature for consistency
    max_tokens: 500   # Limit tokens for speed
    prompt: "Process {input} efficiently"
    system_prompt: "You are an efficient assistant."
    enabled: true     # Explicitly enable/disable

Security Considerations
~~~~~~~~~~~~~~~~~~~~~~

Secure your YAML configurations:

.. code-block:: yaml

    type: llm
    name: secure_assistant
    provider: openai
    model: gpt-4o-mini
    # Don't include sensitive data in YAML
    # api_key: ${OPENAI_API_KEY}  # Use environment variables
    prompt: "Help with {task}"
    system_prompt: "You are a secure assistant."
    validation:
      input_schema:
        fields:
          task: str
          user_id: str
        required_fields: [task, user_id]
      # Validate input to prevent injection attacks

This comprehensive YAML configuration system provides a powerful and flexible way to define agents declaratively while maintaining type safety and validation.
