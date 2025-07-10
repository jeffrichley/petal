AgentBuilder
============

The AgentBuilder provides a fluent interface for building agents using composition with AgentConfig and StepRegistry. This is the underlying implementation that AgentFactory uses internally.

.. automodule:: petal.core.builders.agent
   :members:
   :undoc-members:
   :show-inheritance:

Core Methods
------------

.. method:: AgentBuilder.__init__(state_type: Type)

   Initialize the AgentBuilder with a state type.

   Args:
       state_type: The type for agent state (e.g., TypedDict class)

   Example:
       .. code-block:: python

           from petal.core.builders.agent import AgentBuilder
           from typing import TypedDict

           class MyState(TypedDict):
               messages: list
               name: str

           builder = AgentBuilder(MyState)

.. method:: AgentBuilder.with_step(step_type: str, node_name: Optional[str] = None, **config: Any)

   Add a step to the agent configuration.

   Args:
       step_type: The type of step strategy to use (e.g., "llm", "custom")
       node_name: Optional custom node name for the step
       **config: Configuration parameters for the step

   Returns:
       self: For method chaining

   Example:
       .. code-block:: python

           async def custom_step(state: dict) -> dict:
               state["processed"] = True
               return state

           builder.with_step("custom", step_function=custom_step, node_name="process")

.. method:: AgentBuilder.with_llm(provider: str, model: str, temperature: float = 0.0, max_tokens: int = 8000)

   Add LLM configuration to the agent.

   Args:
       provider: LLM provider (e.g., openai, anthropic, google, cohere, huggingface)
       model: Model name
       temperature: Sampling temperature (0.0 to 2.0, default: 0.0)
       max_tokens: Maximum tokens to generate (default: 8000)

   Returns:
       self: For method chaining

   Example:
       .. code-block:: python

           builder.with_llm(
               provider="openai",
               model="gpt-4o-mini",
               temperature=0.7,
               max_tokens=150
           )

.. method:: AgentBuilder.with_system_prompt(system_prompt: str)

   Add a system prompt to the most recent LLM step.

   Args:
       system_prompt: The system prompt to add to the LLM step

   Returns:
       self: For method chaining

   Example:
       .. code-block:: python

           builder.with_step("llm", prompt_template="Hello {name}")
               .with_system_prompt("You are a {personality} assistant.")

.. method:: AgentBuilder.with_memory(memory_config: Dict[str, Any])

   Add memory configuration to the agent.

   Args:
       memory_config: Dictionary containing memory configuration parameters

   Returns:
       self: For method chaining

   Example:
       .. code-block:: python

           builder.with_memory({
               "memory_type": "conversation",
               "max_tokens": 1000
           })

.. method:: AgentBuilder.with_logging(logging_config: Dict[str, Any])

   Add logging configuration to the agent.

   Args:
       logging_config: Dictionary containing logging configuration parameters

   Returns:
       self: For method chaining

   Example:
       .. code-block:: python

           builder.with_logging({
               "level": "INFO",
               "format": "rich"
           })

.. method:: AgentBuilder.with_graph_config(graph_config: Dict[str, Any])

   Add graph configuration to the agent.

   Args:
       graph_config: Dictionary containing graph configuration parameters

   Returns:
       self: For method chaining

   Example:
       .. code-block:: python

           builder.with_graph_config({
               "topology": "linear",
               "merge_strategy": "append"
           })

.. method:: AgentBuilder.build()

   Build the final agent using AgentBuilderDirector.

   Returns:
       Agent: The built agent ready for execution

   Example:
       .. code-block:: python

           agent = builder.build()
           result = await agent.arun({"name": "Alice", "messages": []})

Complete Example
---------------

.. code-block:: python

    from petal.core.builders.agent import AgentBuilder
    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    # Define custom state
    class CustomState(TypedDict):
        messages: Annotated[list, add_messages]
        name: str
        personality: str
        processed: bool

    # Custom step function
    async def set_personality(state: dict) -> dict:
        state["personality"] = "pirate"
        return state

    # Build agent with multiple steps
    builder = AgentBuilder(CustomState)
    agent = (
        builder.with_step("custom", step_function=set_personality)
        .with_step(
            "llm",
            prompt_template="The user's name is {name}. Say something nice to them."
        )
        .with_system_prompt("You are a {personality} assistant.")
        .with_llm(provider="openai", model="gpt-4o-mini", temperature=0.7)
        .with_memory({"memory_type": "conversation"})
        .build()
    )

    # Run the agent
    result = await agent.arun({
        "name": "Alice",
        "personality": "",
        "processed": False,
        "messages": []
    })
    print(result["messages"][-1].content)

Step Types
----------

The AgentBuilder supports different step types through the StepRegistry:

**LLM Steps**
   Use "llm" as the step_type for language model interactions.

**Custom Steps**
   Use "custom" as the step_type for arbitrary function steps.

**Adding New Step Types**
   You can register new step types by implementing the StepStrategy interface.

Configuration Objects
--------------------

The AgentBuilder uses several configuration objects for different aspects:

.. autoclass:: petal.core.config.agent.AgentConfig
   :members:

.. autoclass:: petal.core.config.agent.StepConfig
   :members:

.. autoclass:: petal.core.config.agent.LLMConfig
   :members:

.. autoclass:: petal.core.config.agent.MemoryConfig
   :members:

.. autoclass:: petal.core.config.agent.GraphConfig
   :members:

.. autoclass:: petal.core.config.agent.LoggingConfig
   :members:
