import asyncio
from unittest.mock import AsyncMock, Mock

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

from petal.core.factory import AgentFactory


class MyModel(BaseModel):
    answer: str


class DefaultState(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    answer: str
    blah: MyModel


async def main():
    # Mock the LLM
    mock_llm = Mock()
    mock_llm.with_structured_output = Mock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(return_value=MyModel(answer="42"))

    from unittest.mock import patch

    import petal.core.steps.llm as llm_mod

    with patch.object(
        llm_mod.LLMStep, "_create_llm_for_provider", return_value=mock_llm
    ):
        agent = (
            AgentFactory(DefaultState)
            .with_chat(prompt_template="What is the answer?", model="gpt-4o-mini")
            .with_structured_output(MyModel)
            .build()
        )
        result = await agent.arun(
            {
                "messages": [HumanMessage(content="test")],
                "name": "test",
                "answer": "old",
                "blah": {"answer": "old"},
            }
        )
        print("Result:", result)
        print("Type of result:", type(result))
        print("Result['answer']:", result.get("answer"))
        print("Result['blah']:", result.get("blah"))

        # Key-wrapped case
        agent2 = (
            AgentFactory(DefaultState)
            .with_chat(prompt_template="What is the answer?", model="gpt-4o-mini")
            .with_structured_output(MyModel, key="blah")
            .build()
        )
        result2 = await agent2.arun(
            {
                "messages": [HumanMessage(content="test")],
                "name": "test",
                "answer": "old",
                "blah": {},
            }
        )
        print("\nKey-wrapped case:")
        print("Result:", result2)
        print("Type of result:", type(result2))
        print("Result['blah']:", result2.get("blah"))
        print("Result['answer']:", result2.get("answer"))


if __name__ == "__main__":
    asyncio.run(main())
