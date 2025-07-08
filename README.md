[![Build Status](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions)
[![Coverage Status](https://coveralls.io/repos/github/<your-username>/<your-repo>/badge.svg?branch=main)](https://coveralls.io/github/<your-username>/<your-repo>?branch=main)
[![PyPI version](https://badge.fury.io/py/petal.svg)](https://badge.fury.io/py/petal)
[![Python Versions](https://img.shields.io/pypi/pyversions/petal.svg)](https://pypi.org/project/petal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pre-commit enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![pip-audit](https://img.shields.io/badge/pip--audit-passing-brightgreen)](https://github.com/pypa/pip-audit)

# 🌸 Petal

**Petal** is an elegant, opinionated agent and tool creation framework for building modular LLM systems using LangChain, LangGraph, and Pydantic.

It’s designed to help you create powerful, discoverable agents and tools with minimal boilerplate — combining declarative structure with fluent chaining.

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

---

## 🚀 Quickstart

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Create an agent with:

```python
from petal import AgentFactory

agent = (
  AgentFactory()
  .with_prompt("Motivate someone who feels {mood}")
  .with_chat()
  .with_tool_registry("my_project.tools")
  .with_logger()
  .build()
)

output = agent.invoke({"mood": "discouraged"})
print(output)
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
