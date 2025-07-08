# Petal Framework – API Specification

## Overview
Petal is an opinionated agent and tool creation framework built on top of LangChain, LangGraph, and Pydantic. It enables fluent chaining of agent steps, automatic tool and LLM resolution, and modular reuse.

...

## Design Notes

- Petal wraps LangGraph and LangChain, providing a higher-level, expressive syntax.
- Agents are composable, testable, and reusable outside LangGraph if needed.
- Framework supports agent creation with or without a dedicated class (e.g., anonymous agents via factory).
