AgentFactory
============

The AgentFactory provides a fluent interface for building agents using LangGraph StateGraphs. It uses the new architecture internally with AgentBuilder composition.

.. automodule:: petal.core.factory
   :members:
   :undoc-members:
   :show-inheritance:

Core Methods
------------

.. method:: AgentFactory.__init__(state_type: type)

   Initialize the AgentFactory with a state type.

   Args:
       state_type: The TypedDict class defining the agent's state schema

   Example:
       .. code-block:: python

           from petal.core.factory import AgentFactory, DefaultState

           factory = AgentFactory(DefaultState)

.. method:: AgentFactory.add(step: Callable, node_name: Optional[str] = None)

   Add a custom step function to the agent.

   Args:
       step: Async function that processes state
       node_name: Optional custom name for the node

   Returns:
       self: For method chaining

   Example:
       .. code-block:: python

           async def custom_step(state: dict) -> dict:
               state["processed"] = True
               return state

           factory.add(custom_step, "process_step")

.. method:: AgentFactory.with_chat(llm=None, prompt_template=None, system_prompt=None, llm_config=None, **kwargs)

   Add an LLM step to the agent chain.

   Args:
       llm: Optional BaseChatModel instance
       prompt_template: Template string for the LLM prompt
       system_prompt: System prompt for the LLM
       llm_config: Dictionary of LLM configuration
       **kwargs: Additional configuration parameters

   Returns:
       self: For method chaining

   Example:
       .. code-block:: python

           factory.with_chat(
               prompt_template="Hello {name}!",
               system_prompt="You are a helpful assistant."
           )

.. method:: AgentFactory.with_prompt(prompt_template: str)

   Set the prompt template for the most recently added LLM step.

   Args:
       prompt_template: Template string with state variable placeholders

   Returns:
       self: For method chaining

   Example:
       .. code-block:: python

           factory.with_chat().with_prompt("Hello {name}!")

.. method:: AgentFactory.with_system_prompt(system_prompt: str)

   Set the system prompt for the most recently added LLM step.

   Args:
       system_prompt: System prompt template with state variable placeholders

   Returns:
       self: For method chaining

   Example:
       .. code-block:: python

           factory.with_chat().with_system_prompt("You are a {personality} assistant.")

.. method:: AgentFactory.build()

   Build the final agent.

   Returns:
       Agent: The built agent ready for execution

   Example:
       .. code-block:: python

           agent = factory.build()
           result = await agent.arun({"name": "Alice"})

Complete Example
---------------

.. code-block:: python

    from petal.core.factory import AgentFactory, DefaultState

    # Create agent with custom step and LLM
    async def process_name(state: dict) -> dict:
        state["greeting"] = f"Hello {state['name']}!"
        return state

    agent = (
        AgentFactory(DefaultState)
        .add(process_name)
        .with_chat(
            prompt_template="The user's name is {name}. Say something nice.",
            system_prompt="You are a friendly assistant."
        )
        .build()
    )

    # Run the agent
    result = await agent.arun({"name": "Alice", "messages": []})
    print(result["messages"][-1].content)

State Types
-----------

The framework provides several predefined state types:

.. autoclass:: petal.core.factory.DefaultState
   :members:

.. autoclass:: petal.core.factory.NonChatState
   :members:

.. autoclass:: petal.core.factory.MergeableState
   :members:

You can also define custom state types using TypedDict:

.. code-block:: python

    from typing import Annotated, TypedDict
    from langgraph.graph.message import add_messages

    class CustomState(TypedDict):
        messages: Annotated[list, add_messages]
        user_input: str
        response: str
        metadata: dict
