[![CI](https://github.com/jeffrichley/petal/actions/workflows/ci.yml/badge.svg)](https://github.com/jeffrichley/petal/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/petal.svg)](https://badge.fury.io/py/petal)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pre-commit enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![pip-audit](https://img.shields.io/badge/pip--audit-passing-brightgreen)](https://github.com/pypa/pip-audit)
[![Coverage](https://codecov.io/gh/jeffrichley/petal/branch/main/graph/badge.svg)](https://codecov.io/gh/jeffrichley/petal)

# 🌸 Petal

**Petal** is an elegant, opinionated agent and tool creation framework for building modular LLM systems using LangChain, LangGraph, and Pydantic.

It's designed to help you create powerful, discoverable agents and tools with minimal boilerplate — combining declarative structure with fluent chaining.

---

## ✨ Features

- 🔗 **Fluent Chaining API**
  Compose agents with readable, chainable setup flows (`.with_chat()`, `.with_prompt()`, etc.)

- 🧠 **Tool & MCP Discovery**
  Automatically register local and external tools using a configurable registry.

- 🏗️ **LangGraph Integration**
  Agents are directly runnable as LangGraph nodes with native support for state merging and memory.

- ⚙️ **Factory-Based Architecture**
  Easily scaffold agents and tools using expressive factory methods with IDE support.

- 📄 **Rich Manifest Typing**
  All agents and tools are strongly typed using `TypedDict` for autocompletion and clarity.

- 🔁 **Declarative State Merging**
  Fields can auto-merge (e.g., `append`, `extend`) instead of overwriting.

- 🎯 **Named Parameter LLM Configuration**
  Use `with_llm(provider, model, temperature=0.0)` instead of magic dictionary keys.

- 🏠 **Local LLM Support**
  Run local models via Ollama with the same interface as cloud providers.

- 💬 **System Prompt Support**
  Add system prompts with state variable interpolation for dynamic behavior.

---

## 🚀 Quickstart

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Local LLM Setup (Optional)

To use local LLMs via Ollama:

1. **Install Ollama**: [https://ollama.ai/](https://ollama.ai/)
2. **Pull a model**: `ollama pull llama2`
3. **Start Ollama**: `ollama serve`
4. **Test the demo**: `python examples/ollama_demo.py`

### Simple Agent with AgentFactory

```python
from petal.core.factory import AgentFactory, DefaultState

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
```

### Local LLM with Ollama

```python
from petal.core.factory import AgentFactory, DefaultState

# Use local LLM via Ollama
agent = (
    AgentFactory(DefaultState)
    .with_chat(
        prompt_template="Hello {name}! How can I help you today?",
        system_prompt="You are a helpful and friendly assistant.",
        llm_config={
            "provider": "ollama",
            "model": "llama2",
            "temperature": 0.7,
        }
    )
    .build()
)

result = await agent.arun({"name": "Alice", "messages": []})
print(result["messages"][-1].content)
```

### Advanced Agent with Custom Steps

```python
from petal.core.factory import AgentFactory
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class CustomState(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    personality: str

async def set_personality(state: dict) -> dict:
    state["personality"] = "pirate"
    return state

agent = (
    AgentFactory(CustomState)
    .add(set_personality)
    .with_chat(
        prompt_template="The user's name is {name}. Say something nice to them.",
        system_prompt="You are a {personality} assistant."
    )
    .build()
)

result = await agent.arun({
    "name": "Alice",
    "personality": "",
    "messages": []
})
print(result["messages"][-1].content)
```

### Using AgentBuilder (Lower-level API)

```python
from petal.core.builders.agent import AgentBuilder
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str

builder = AgentBuilder(MyState)
agent = (
    builder.with_step(
        "llm",
        prompt_template="User says: {user_input}. Respond helpfully."
    )
    .with_system_prompt("You are a helpful assistant.")
    .with_llm(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=150
    )
    .build()
)

result = await agent.arun({
    "user_input": "Hello! How are you today?",
    "messages": []
})
print(result["messages"][-1].content)
```

---

## 🚀 Quickstart for Developers

1. Clone this repository:
   ```sh
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Run the automated setup script:
   ```sh
   python scripts/setup_dev.py
   ```
   - This will create a Python 3.11 virtual environment, install all dependencies, set up pre-commit hooks, and run initial checks.
3. Activate your virtual environment:
   - **Windows PowerShell:**
     ```sh
     .venv\Scripts\Activate.ps1
     ```
   - **Windows CMD:**
     ```sh
     .venv\Scripts\activate.bat
     ```
   - **Mac/Linux:**
     ```sh
     source .venv/bin/activate
     ```
4. Start coding! Pre-commit hooks and all dev tools are ready to go.

For more details, see CONTRIBUTING.md and the rest of this README.

---

## 🧰 Project Structure

```
petal/
├── src/petal/
│   ├── core/
│   │   ├── factory.py          # AgentFactory - High-level API
│   │   ├── builders/
│   │   │   ├── agent.py        # AgentBuilder - Lower-level API
│   │   │   └── director.py     # AgentBuilderDirector
│   │   ├── config/
│   │   │   ├── agent.py        # Configuration objects
│   │   │   └── state.py        # State type factory
│   │   ├── steps/
│   │   │   ├── base.py         # StepStrategy ABC
│   │   │   ├── llm.py          # LLMStepStrategy
│   │   │   ├── custom.py       # CustomStepStrategy
│   │   │   └── registry.py     # StepRegistry
│   │   └── tool_factory.py     # ToolFactory
│   └── types/
│       ├── agent_manifest.py   # Agent type definitions
│       └── tool_manifest.py    # Tool type definitions
├── examples/
│   ├── playground.py           # Basic AgentFactory usage
│   ├── playground2.py          # Rich logging example
│   ├── improved_api_demo.py    # AgentBuilder usage
│   └── custom_tool.py          # Custom tool example
└── tests/
    └── petal/
        ├── test_factory.py     # AgentFactory tests
        ├── test_builders_agent.py  # AgentBuilder tests
        └── test_steps_*.py     # Step strategy tests
```

---

## 📚 Documentation

- [Getting Started](docs/source/getting_started.rst) - Quick start guide
- [API Reference](docs/source/api/index.rst) - Complete API documentation
- [Examples](docs/source/examples/index.rst) - Tutorials and examples
- [Architecture](docs/source/architecture.rst) - Framework architecture overview

---

## 🎯 Key Features

### AgentFactory (High-level API)
- Fluent interface for quick agent creation
- Automatic step management
- Backward compatibility with existing code
- Simple state type handling

### AgentBuilder (Lower-level API)
- Explicit step configuration
- Named parameter LLM configuration
- Direct access to configuration objects
- More control over the building process

### State Management
- Strongly-typed state with TypedDict
- Automatic message handling with `add_messages`
- State variable interpolation in prompts
- System prompt formatting with state variables

### LLM Integration
- Support for multiple providers (OpenAI, Anthropic, etc.)
- Named parameter configuration for better IDE support
- System prompts with state variable formatting
- Automatic message handling and state management

---

## 🧪 Testing

Run the test suite:

```bash
uv run make test
```

Run with coverage:

```bash
uv run make coverage
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Consistent Formatting, Linting, and Type Checking

- Run `make checkit` to check types, run pre-commit hooks (formatting, linting, type checking), and run tests with coverage.
- If pre-commit makes changes, stage them (`git add .`) and re-commit.
- Optionally, run `make precommit-autofix` to auto-fix and stage all changes before committing.
