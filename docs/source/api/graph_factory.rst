GraphFactory
============

Factory for building LangGraph workflows and managing graph composition.

.. automodule:: petal.core.graph_factory
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The GraphFactory provides utilities for building and managing LangGraph workflows. It enables composition of multiple agents and steps into complex graph structures with support for conditional edges, parallel execution, and state management.

Key Features
------------

- **Graph Composition**: Combine multiple agents and steps into workflows
- **Conditional Edges**: Dynamic routing based on state conditions
- **Parallel Execution**: Support for concurrent step execution
- **State Management**: Automatic state merging and validation
- **Edge Configuration**: Flexible edge definition and routing
- **Visualization**: Graph diagram generation and export
- **Validation**: Graph structure validation and error detection

Basic Usage
-----------

.. code-block:: python

    from petal.core.graph_factory import GraphFactory
    from petal.core.factory import AgentFactory, DefaultState

    # Create individual agents
    agent1 = await AgentFactory(DefaultState).with_chat("Step 1").build()
    agent2 = await AgentFactory(DefaultState).with_chat("Step 2").build()

    # Create graph factory
    graph_factory = GraphFactory()

    # Build workflow
    workflow = (
        graph_factory
        .add_agent("step1", agent1)
        .add_agent("step2", agent2)
        .connect("step1", "step2")
        .build()
    )

    # Run workflow
    result = await workflow.ainvoke({"messages": []})

Advanced Workflow with Conditions
--------------------------------

.. code-block:: python

    from petal.core.graph_factory import GraphFactory
    from petal.core.factory import AgentFactory, DefaultState

    # Create agents for different paths
    analysis_agent = await AgentFactory(DefaultState).with_chat("Analyze").build()
    action_agent = await AgentFactory(DefaultState).with_chat("Take action").build()
    review_agent = await AgentFactory(DefaultState).with_chat("Review").build()

    # Define condition function
    def should_take_action(state):
        return state.get("needs_action", False)

    # Build conditional workflow
    workflow = (
        GraphFactory()
        .add_agent("analyze", analysis_agent)
        .add_agent("action", action_agent)
        .add_agent("review", review_agent)
        .connect("analyze", "action", condition=should_take_action)
        .connect("analyze", "review", condition=lambda s: not should_take_action(s))
        .connect("action", "review")
        .build()
    )

Parallel Execution
-----------------

.. code-block:: python

    from petal.core.graph_factory import GraphFactory
    from petal.core.factory import AgentFactory, DefaultState

    # Create parallel agents
    agent_a = await AgentFactory(DefaultState).with_chat("Task A").build()
    agent_b = await AgentFactory(DefaultState).with_chat("Task B").build()
    agent_c = await AgentFactory(DefaultState).with_chat("Task C").build()

    # Build parallel workflow
    workflow = (
        GraphFactory()
        .add_agent("task_a", agent_a)
        .add_agent("task_b", agent_b)
        .add_agent("task_c", agent_c)
        .add_agent("combine", combine_agent)
        .connect_parallel(["task_a", "task_b", "task_c"], "combine")
        .build()
    )

Method Reference
----------------

.. method:: GraphFactory.__init__() -> None

    Initialize the graph factory.

.. method:: GraphFactory.add_agent(name: str, agent: Agent) -> GraphFactory

    Add an agent to the graph.

    Args:
        name: Name for the agent node
        agent: The agent instance to add

    Returns:
        self for chaining

    Example:
        .. code-block:: python

            agent = await AgentFactory(DefaultState).with_chat("Hello").build()
            graph_factory.add_agent("greeter", agent)

.. method:: GraphFactory.add_step(name: str, step: Callable) -> GraphFactory

    Add a custom step function to the graph.

    Args:
        name: Name for the step node
        step: The step function to add

    Returns:
        self for chaining

    Example:
        .. code-block:: python

            async def custom_step(state):
                return {"processed": True}

            graph_factory.add_step("process", custom_step)

.. method:: GraphFactory.connect(from_node: str, to_node: str, condition: Optional[Callable] = None) -> GraphFactory

    Connect two nodes in the graph.

    Args:
        from_node: Source node name
        to_node: Target node name
        condition: Optional condition function for conditional edges

    Returns:
        self for chaining

    Example:
        .. code-block:: python

            # Simple connection
            graph_factory.connect("step1", "step2")

            # Conditional connection
            def should_continue(state):
                return state.get("continue", True)

            graph_factory.connect("step1", "step2", condition=should_continue)

.. method:: GraphFactory.connect_parallel(from_nodes: List[str], to_node: str) -> GraphFactory

    Connect multiple nodes in parallel to a single target node.

    Args:
        from_nodes: List of source node names
        to_node: Target node name

    Returns:
        self for chaining

    Example:
        .. code-block:: python

            graph_factory.connect_parallel(["task_a", "task_b", "task_c"], "combine")

.. method:: GraphFactory.set_entry_point(node: str) -> GraphFactory

    Set the entry point for the graph.

    Args:
        node: Name of the entry point node

    Returns:
        self for chaining

    Example:
        .. code-block:: python

            graph_factory.set_entry_point("start")

.. method:: GraphFactory.set_exit_point(node: str) -> GraphFactory

    Set the exit point for the graph.

    Args:
        node: Name of the exit point node

    Returns:
        self for chaining

    Example:
        .. code-block:: python

            graph_factory.set_exit_point("end")

.. method:: GraphFactory.build() -> StateGraph

    Build the final graph.

    Returns:
        The compiled StateGraph instance

    Raises:
        ValueError: If graph structure is invalid

    Example:
        .. code-block:: python

            workflow = graph_factory.build()
            result = await workflow.ainvoke(initial_state)

.. method:: GraphFactory.diagram(path: str, format: str = "png") -> None

    Generate a diagram of the graph.

    Args:
        path: File path for the diagram
        format: Output format (png, svg, pdf)

    Example:
        .. code-block:: python

            graph_factory.diagram("workflow.png")

Error Handling
--------------

The GraphFactory provides comprehensive error handling:

**Invalid Node Names**
    .. code-block:: python

        try:
            graph_factory.connect("nonexistent", "target")
        except ValueError as e:
            print(f"Node not found: {e}")

**Circular Dependencies**
    .. code-block:: python

        try:
            graph_factory.connect("a", "b").connect("b", "a")
            workflow = graph_factory.build()
        except ValueError as e:
            print(f"Circular dependency detected: {e}")

**Missing Entry/Exit Points**
    .. code-block:: python

        try:
            workflow = graph_factory.build()
        except ValueError as e:
            print(f"Graph structure invalid: {e}")

Integration with AgentFactory
----------------------------

The GraphFactory integrates seamlessly with AgentFactory:

.. code-block:: python

    from petal.core.graph_factory import GraphFactory
    from petal.core.factory import AgentFactory, DefaultState

    # Create agents using AgentFactory
    analysis_agent = await (
        AgentFactory(DefaultState)
        .with_chat("Analyze the input")
        .with_tools(["search", "calculator"])
        .build()
    )

    action_agent = await (
        AgentFactory(DefaultState)
        .with_chat("Take action based on analysis")
        .with_react_loop(["filesystem", "email"])
        .build()
    )

    # Compose into workflow
    workflow = (
        GraphFactory()
        .add_agent("analyze", analysis_agent)
        .add_agent("action", action_agent)
        .connect("analyze", "action")
        .build()
    )

    # Run workflow
    result = await workflow.ainvoke({
        "messages": [],
        "task": "Process user request"
    })

Performance Considerations
-------------------------

- **Lazy Compilation**: Graphs are compiled only when build() is called
- **Efficient Routing**: Conditional edges use optimized routing logic
- **State Merging**: Automatic state merging for parallel execution
- **Memory Management**: Efficient memory usage for large graphs

Thread Safety
-------------

The GraphFactory is designed for concurrent access:

- **Immutable Configuration**: Graph configuration is thread-safe
- **Compiled Graphs**: Built graphs are safe for concurrent execution
- **State Isolation**: Each execution has isolated state
