from typing import List, Optional

import pytest
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from petal.core.steps.llm import LLMStep, LLMStepStrategy


class DummyLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "dummy"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        # Silence unused argument warnings
        _ = messages
        _ = stop
        _ = run_manager
        _ = kwargs
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="Hello!", response_metadata={})
                )
            ]
        )


@pytest.mark.asyncio
async def test_llm_step_strategy_create_and_call():
    print("A: before strategy")
    strategy = LLMStepStrategy()
    print("B: before config")
    config = {
        "prompt_template": "Hello {name}",
        "system_prompt": "You are a helpful assistant.",
        "llm_config": {"model": "gpt-4o-mini"},
        "llm_instance": DummyLLM(),
    }
    print("C: before create_step")
    step = strategy.create_step(config)
    print("D: before isinstance")
    assert isinstance(step, LLMStep)
    print("E: before get_node_name")
    node_name = strategy.get_node_name(0)
    print("F: before assert node_name")
    assert node_name == "llm_step_0"
    print("G: before state")
    state = {"name": "Jeff", "messages": []}
    print("H: before await step")
    result = await step(state)
    print("I: before assert messages in result")
    assert "messages" in result
    print("J: before isinstance messages[0]")
    assert isinstance(result["messages"][0], dict)
    print("K: before assert role user")
    assert result["messages"][0]["role"] == "user"
    print("L: before isinstance messages[1]")
    assert isinstance(result["messages"][1], AIMessage)
    print("M: before assert type ai")
    assert result["messages"][1].type == "ai"
    print("N: before assert content Hello!")
    assert result["messages"][1].content == "Hello!"


def test_llm_step_strategy_config_validation():
    strategy = LLMStepStrategy()
    # Missing llm_instance and llm_config
    config = {"prompt_template": "Hi"}
    with pytest.raises(ValueError):
        strategy.create_step(config)


def test_llm_step_strategy_node_name():
    strategy = LLMStepStrategy()
    assert strategy.get_node_name(5) == "llm_step_5"
