AgentFactory
============

The main high-level API for building agents with fluent chaining, tool integration, React loops, YAML loading, and diagram generation.

.. automodule:: petal.core.factory
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

AgentFactory provides a fluent, chainable interface for creating AI agents. It encapsulates the complexity of agent building while providing access to all major features:

- **LLM Integration**: Chat steps with prompt templates and system prompts
- **Custom Steps**: Arbitrary functions that process state
- **Tool Integration**: Automatic tool discovery and binding
- **ReAct Loops**: Reasoning and tool use workflows
- **YAML Configuration**: Declarative agent definitions
- **Structured Output**: Pydantic model binding
- **Tool Discovery**: Configurable discovery strategies
- **Diagram Generation**: Visual representation of agents and graphs

Basic Usage
-----------

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState

    # Simple agent with LLM
    agent = (
        AgentFactory(DefaultState)
        .with_chat(
            prompt_template="Hello {name}! How can I help you today?",
            system_prompt="You are a helpful and friendly assistant."
        )
        .build()
    )

    result = await agent.arun({"name": "Alice", "messages": []})
    print(result["messages"][-1].content)

Advanced Usage
--------------

.. code-block:: python

    from petal.core.factory import AgentFactory
    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages
    from pydantic import BaseModel

    class CustomState(TypedDict):
        messages: Annotated[list, add_messages]
        name: str
        personality: str
        task: str

    class TaskResult(BaseModel):
        completed: bool
        result: str
        confidence: float

    async def set_personality(state: dict) -> dict:
        state["personality"] = "pirate"
        return state

    agent = (
        AgentFactory(CustomState)
        .add(set_personality)
        .with_chat(
            prompt_template="Complete this task: {task}",
            system_prompt="You are a {personality} assistant."
        )
        .with_structured_output(TaskResult)
        .with_tools(["calculator:add", "web_search"])
        .with_tool_discovery(
            enabled=True,
            folders=["tools/"],
            config_locations=["config/tools.yaml"]
        )
        .build()
    )

    result = await agent.arun({
        "name": "Captain",
        "personality": "",
        "task": "Calculate 15 + 27",
        "messages": []
    })

Method Reference
----------------

.. method:: AgentFactory.__init__(state_type: type)

    Initialize the AgentFactory with a state type.

    Args:
        state_type: The TypedDict class defining the agent's state schema

    Raises:
        TypeError: If state_type is None

.. method:: AgentFactory.add(step: Callable[..., Any], node_name: Optional[str] = None) -> AgentFactory

    Add a custom step function to the agent.

    Args:
        step: The function to add as a step
        node_name: Optional custom name for the step node

    Returns:
        self for fluent chaining

.. method:: AgentFactory.with_chat(llm: Optional[Any] = None, prompt_template: Optional[str] = None, system_prompt: Optional[str] = None, llm_config: Optional[Dict[str, Any] | LLMConfig] = None, **kwargs) -> AgentFactory

    Add an LLM chat step to the agent.

    Args:
        llm: Optional pre-configured LLM instance
        prompt_template: Template string for user prompts
        system_prompt: System prompt for the LLM
        llm_config: LLM configuration as dict or LLMConfig object
        **kwargs: Additional configuration parameters

    Returns:
        self for fluent chaining

    Raises:
        ValueError: If llm is not a BaseChatModel instance

.. method:: AgentFactory.with_prompt(prompt_template: str) -> AgentFactory

    Set the prompt template for the most recent LLM or React step.

    Args:
        prompt_template: The prompt template string

    Returns:
        self for fluent chaining

    Raises:
        ValueError: If no steps have been added or the most recent step is not LLM/React

.. method:: AgentFactory.with_system_prompt(system_prompt: str) -> AgentFactory

    Set the system prompt for the most recent LLM or React step.

    Args:
        system_prompt: The system prompt string

    Returns:
        self for fluent chaining

    Raises:
        ValueError: If no steps have been added or the most recent step is not LLM/React

.. method:: AgentFactory.with_structured_output(model: Any, key: Optional[str] = None) -> AgentFactory

    Bind a structured output schema to the most recent LLM or React step.

    Args:
        model: Pydantic model class for structured output
        key: Optional key to wrap the output in a dict

    Returns:
        self for fluent chaining

.. method:: AgentFactory.with_tools(tools: List[Union[str, Any]], scratchpad_key: Optional[str] = None) -> AgentFactory

    Add tools to the most recent LLM step.

    Args:
        tools: List of tool names (strings) or tool objects
        scratchpad_key: Optional key for storing tool observations

    Returns:
        self for fluent chaining

    Raises:
        ValueError: If no steps have been added, the most recent step is not LLM, or tools list is empty

.. method:: AgentFactory.with_react_tools(tools: List[Union[str, Any]], scratchpad_key: str = "scratchpad") -> AgentFactory

    Add tools with ReAct-style scratchpad for persistent tool observation history.

    Args:
        tools: List of tool names (strings) or tool objects
        scratchpad_key: Key for storing tool observations (defaults to "scratchpad")

    Returns:
        self for fluent chaining

.. method:: AgentFactory.with_react_loop(tools: List[Union[str, Any]], **config) -> AgentFactory

    Add a ReAct reasoning loop step.

    Args:
        tools: List of tool names (strings) or tool objects
        **config: Additional configuration for the React step

    Returns:
        self for fluent chaining

.. method:: AgentFactory.with_tool_discovery(enabled: bool = True, folders: Optional[List[str]] = None, config_locations: Optional[List[str]] = None, exclude_patterns: Optional[List[str]] = None) -> AgentFactory

    Configure tool discovery for this agent factory.

    Args:
        enabled: Whether to enable tool discovery
        folders: List of folders to scan for tools
        config_locations: List of config file locations to scan
        exclude_patterns: List of patterns to exclude from discovery

    Returns:
        self for fluent chaining

.. method:: AgentFactory.build() -> Agent

    Build the agent from the current configuration.

    Returns:
        The built Agent instance

    Raises:
        ValueError: If no steps have been configured

.. method:: AgentFactory.node_from_yaml(path: str)

    Load a node configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        self for fluent chaining

    Raises:
        YAMLFileNotFoundError: If the YAML file is not found
        YAMLParseError: If the YAML file has invalid syntax
        ValueError: If the node type is unsupported

.. method:: AgentFactory.diagram_agent(agent: "Agent", path: str, format: str = "png") -> None

    Generate a diagram of the agent's graph structure.

    Args:
        agent: The agent to diagram
        path: Output file path
        format: Output format (png, svg, etc.)

    Raises:
        ValueError: If the agent is not built or format is unsupported

.. method:: AgentFactory.diagram_graph(path: str, format: str = "png") -> Any

    Generate a diagram of the current graph configuration.

    Args:
        path: Output file path
        format: Output format (png, svg, etc.)

    Returns:
        The graph object

    Raises:
        ValueError: If no steps have been configured

State Types
-----------

The framework provides several built-in state types:

.. class:: DefaultState

    Default state schema with messages and name fields.

    .. code-block:: python

        class DefaultState(TypedDict):
            messages: Annotated[list, add_messages]
            name: str

You can also define custom state types:

.. code-block:: python

    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class CustomState(TypedDict):
        messages: Annotated[list, add_messages]
        user_input: str
        response: str
        metadata: dict

LLM Configuration
-----------------

LLM configuration can be provided in several ways:

**Direct LLM Instance**
    .. code-block:: python

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini")
        agent.with_chat(llm=llm)

**Configuration Dictionary**
    .. code-block:: python

        agent.with_chat(
            llm_config={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )

**LLMConfig Object**
    .. code-block:: python

        from petal.core.config.agent import LLMConfig

        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000
        )
        agent.with_chat(llm_config=config)

Supported Providers
~~~~~~~~~~~~~~~~~~

- **openai**: OpenAI models (GPT-4, GPT-3.5, etc.)
- **anthropic**: Anthropic models (Claude)
- **google**: Google models (Gemini)
- **cohere**: Cohere models
- **huggingface**: Hugging Face models
- **ollama**: Local models via Ollama

Tool Integration
----------------

Tools can be integrated in several ways:

**String Names**
    Tools are resolved by name from the registry:
    .. code-block:: python

        agent.with_tools(["calculator:add", "web_search"])

**Tool Objects**
    Direct tool objects can be provided:
    .. code-block:: python

        from petal.core.decorators import petaltool

        @petaltool("calculator:add")
        def add(a: float, b: float) -> float:
            return a + b

        agent.with_tools([add])

**MCP Tools**
    MCP tools are automatically resolved:
    .. code-block:: python

        agent.with_tools(["mcp:filesystem:read_file"])

Tool Discovery
--------------

Tool discovery can be configured with multiple strategies:

.. code-block:: python

    agent.with_tool_discovery(
        enabled=True,
        folders=["tools/", "my_tools/"],
        config_locations=["config/tools.yaml"],
        exclude_patterns=["*_test.py", "*.pyc"]
    )

Discovery Strategies
~~~~~~~~~~~~~~~~~~~

- **Decorator Discovery**: Finds `@petaltool` decorated functions
- **Config Discovery**: Loads tools from YAML configuration files
- **Folder Discovery**: Scans directories for Python files
- **MCP Discovery**: Discovers tools from MCP servers

Error Handling
--------------

The AgentFactory provides comprehensive error handling:

**Configuration Errors**
    Invalid configurations are caught early with clear error messages.

**Tool Resolution Errors**
    Missing or ambiguous tools are handled gracefully.

**YAML Parsing Errors**
    Invalid YAML files are caught with detailed error information.

**Build Errors**
    Missing or invalid step configurations are caught during build.

Examples
--------

**Simple Chat Agent**
    .. code-block:: python

        agent = (
            AgentFactory(DefaultState)
            .with_chat(
                prompt_template="Hello {name}! How can I help you today?",
                system_prompt="You are a helpful and friendly assistant."
            )
            .build()
        )

**Agent with Tools**
    .. code-block:: python

        agent = (
            AgentFactory(DefaultState)
            .with_chat(
                prompt_template="Calculate {expression} for me.",
                system_prompt="You are a helpful math assistant."
            )
            .with_tools(["calculator:add", "calculator:multiply"])
            .build()
        )

**ReAct Agent**
    .. code-block:: python

        agent = (
            AgentFactory(DefaultState)
            .with_react_loop(
                tools=["calculator:add", "web_search"],
                reasoning_prompt="Think step by step about how to solve this.",
                system_prompt="You are a helpful assistant that can use tools."
            )
            .build()
        )

**Agent with Structured Output**
    .. code-block:: python

        from pydantic import BaseModel

        class MathResult(BaseModel):
            answer: float
            method: str
            confidence: float

        agent = (
            AgentFactory(DefaultState)
            .with_chat(
                prompt_template="Solve {problem} and explain your method.",
                system_prompt="You are a math tutor."
            )
            .with_structured_output(MathResult)
            .build()
        )

**Agent from YAML**
    .. code-block:: python

        # agent.yaml
        type: llm
        name: assistant
        provider: openai
        model: gpt-4o-mini
        prompt: "Help with {task}"
        system_prompt: "You are a helpful assistant."

        agent = (
            AgentFactory(DefaultState)
            .node_from_yaml("agent.yaml")
            .build()
        )

**Agent with Custom Steps**
    .. code-block:: python

        async def preprocess(state: dict) -> dict:
            state["processed_input"] = state["input"].lower()
            return state

        async def postprocess(state: dict) -> dict:
            state["final_result"] = f"Processed: {state['response']}"
            return state

        agent = (
            AgentFactory(CustomState)
            .add(preprocess)
            .with_chat(
                prompt_template="Process: {processed_input}",
                system_prompt="You are a helpful processor."
            )
            .add(postprocess)
            .build()
        )
