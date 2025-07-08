from petal.core.factory import AgentFactory


def step1(state):
    state.setdefault("messages", []).append("User: " + state["input"])
    return state


def step2(state):
    state["messages"].append("LLM: " + state["messages"][-1].upper())
    return state


if __name__ == "__main__":
    factory = AgentFactory()
    agent = (
        factory.add(step1)
        .add(step2)
        .with_prompt("Echo: {input}")
        .with_system_prompt("You are an echo bot.")
        .build()
    )
    state = {"input": "hello", "messages": []}
    out = agent.run(state)
    print(out["messages"])
    # Output: ["User: hello", "LLM: USER: HELLO"]
