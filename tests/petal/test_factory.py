import pytest
from petal.core.factory import AgentFactory


def test_agent_factory_init() -> None:
    af = AgentFactory()
    assert isinstance(af, AgentFactory)


def test_agent_factory_normal():
    def step1(state):
        state["x"] = 1
        return state

    def step2(state):
        state["x"] += 2
        return state

    agent = (
        AgentFactory()
        .add(step1)
        .add(step2)
        .with_prompt("Test {x}")
        .with_system_prompt("System")
        .build()
    )
    result = agent.run({})
    assert result["x"] == 3


def test_agent_factory_no_steps():
    factory = AgentFactory()
    with pytest.raises(RuntimeError):
        factory.build()


def test_agent_run_before_build():
    from petal.core.factory import Agent

    agent = Agent([], "", "")
    agent.built = False
    with pytest.raises(RuntimeError):
        agent.run({})
