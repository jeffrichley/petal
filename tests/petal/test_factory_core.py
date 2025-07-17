"""Core AgentFactory tests for basic agent lifecycle and functionality."""

from typing import Any, Dict

import pytest
from langgraph.graph import END, START, StateGraph
from petal.core.agent import Agent
from petal.core.factory import AgentFactory
from typing_extensions import TypedDict

from tests.petal.conftest_factory import SimpleState


@pytest.mark.asyncio
async def test_agent_factory_normal():
    async def step1(_state):
        return {"x": 1}

    async def step2(state):
        return {"x": state.get("x", 0) + 2}

    agent = await AgentFactory(SimpleState).add(step1).add(step2).build()
    result = await agent.arun({})
    assert result["x"] == 3


@pytest.mark.asyncio
async def test_agent_factory_no_steps():
    factory = AgentFactory(SimpleState)
    with pytest.raises(ValueError, match="Cannot build agent: no steps configured"):
        await factory.build()


@pytest.mark.asyncio
async def test_agent_arun_before_build():
    agent = Agent()
    with pytest.raises(RuntimeError):
        await agent.arun({})


@pytest.mark.asyncio
async def test_agent_arun_with_none_graph():
    agent = Agent()
    agent.built = True
    agent.graph = None
    with pytest.raises(
        RuntimeError, match="Agent.graph is None - agent was not properly built"
    ):
        await agent.arun({})


@pytest.mark.asyncio
async def test_agent_build_method():
    class TestState(TypedDict):
        test: str

    graph = StateGraph(TestState)
    graph.add_node("test", lambda x: x)
    graph.add_edge(START, "test")
    graph.add_edge("test", END)
    compiled_graph = graph.compile()
    agent = Agent().build(compiled_graph, TestState)
    assert agent.built is True
    assert agent.graph is compiled_graph
    assert agent.state_type is TestState
    result = await agent.arun({"test": "value"})
    assert result == {"test": "value"}


@pytest.mark.asyncio
async def test_agent_with_custom_node_names():
    async def step1(_state):
        return {"x": 1}

    async def step2(state):
        return {"x": state.get("x", 0) + 2}

    agent = await (
        AgentFactory(SimpleState)
        .add(step1, "custom_step_1")
        .add(step2, "custom_step_2")
        .build()
    )
    result = await agent.arun({})
    assert result["x"] == 3


@pytest.mark.asyncio
async def test_agent_with_typed_dict_state():
    class MyState(TypedDict):
        messages: list
        processed: bool

    async def step(_state: MyState) -> Dict[str, Any]:
        return {"processed": True}

    agent = await AgentFactory(state_type=MyState).add(step).build()
    result = await agent.arun({"messages": [], "processed": False})
    assert result["processed"] is True
    assert "messages" in result


@pytest.mark.asyncio
async def test_agent_build_with_single_step():
    async def step(_state):
        return {"processed": True}

    agent = await AgentFactory(SimpleState).add(step).build()
    result = await agent.arun({})
    assert result["processed"] is True


@pytest.mark.asyncio
async def test_agent_build_with_no_steps_raises():
    factory = AgentFactory(SimpleState)
    with pytest.raises(ValueError, match="Cannot build agent: no steps configured"):
        await factory.build()


@pytest.mark.asyncio
async def test_agent_arun_with_built_flag():
    agent = await AgentFactory(SimpleState).add(lambda _: {"processed": True}).build()
    result = await agent.arun({})
    assert result["processed"] is True


@pytest.mark.asyncio
async def test_agent_factory_uses_new_architecture_internally():
    from petal.core.builders.agent import AgentBuilder

    factory = AgentFactory(SimpleState)
    assert hasattr(factory, "_builder")
    assert isinstance(factory._builder, AgentBuilder)
    assert factory._builder._config.state_type == SimpleState

    async def test_step(_state):
        return {"x": 1}

    factory.add(test_step)
    assert len(factory._builder._config.steps) == 1
    assert factory._builder._config.steps[0].strategy_type == "custom"
    assert factory._builder._config.steps[0].config["step_function"] == test_step
    agent = await factory.build()
    assert agent is not None
    assert hasattr(agent, "arun")
