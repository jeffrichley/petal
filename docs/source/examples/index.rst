Examples & Tutorials
====================

This section contains tutorials and examples showing how to use Petal for various tasks.

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   basic_agent
   tool_integration
   graph_workflows
   custom_tools
   memory_management

Basic Examples
--------------

Simple Agent
~~~~~~~~~~~~

Create a basic agent with a single tool:

.. code-block:: python

   from petal import AgentFactory, tool_fn

   @tool_fn
   def greet(name: str) -> str:
       """Greet someone by name."""
       return f"Hello, {name}!"

   agent = (AgentFactory()
       .add(greet)
       .with_prompt("Greet {name}")
       .with_chat()
       .build())

   result = agent.run({"name": "World"})
   print(result)  # "Hello, World!"

Tool Integration
~~~~~~~~~~~~~~~

Combine multiple tools in a single agent:

.. code-block:: python

   from petal import AgentFactory, tool_fn

   @tool_fn
   def get_user_info(user_id: str) -> dict:
       """Get user information."""
       return {"name": "John", "email": "john@example.com"}

   @tool_fn
   def send_email(to: str, subject: str, body: str) -> bool:
       """Send an email."""
       return True

   agent = (AgentFactory()
       .add(get_user_info)
       .add(send_email)
       .with_prompt("Send a welcome email to user {user_id}")
       .with_chat()
       .build())

Advanced Examples
-----------------

Graph Workflows
~~~~~~~~~~~~~~~

Create complex workflows with multiple agents:

.. code-block:: python

   from petal import GraphFactory, AgentFactory

   # Create specialized agents
   research_agent = (AgentFactory()
       .with_prompt("Research {topic}")
       .with_chat()
       .build())

   write_agent = (AgentFactory()
       .with_prompt("Write about {topic} using {research}")
       .with_chat()
       .build())

   # Compose into a workflow
   workflow = (GraphFactory()
       .add_agent("research", research_agent)
       .add_agent("write", write_agent)
       .connect("research", "write")
       .build())

   result = workflow.run({"topic": "AI agents"})
