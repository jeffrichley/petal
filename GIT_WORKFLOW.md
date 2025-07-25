# Git Workflow Guide

**IMPORTANT:**
All commands in this guide use [`uv`](https://github.com/astral-sh/uv) to ensure they run inside your project's virtual environment, regardless of your platform (Mac, Linux, or Windows).

- No need to manually activate your virtual environment—just use `uv run <command>` as shown below.

---

## Project Structure

The codebase follows a modular architecture with clear separation of concerns:

### Core Modules
- `src/petal/core/factory.py` - Public API (AgentFactory)
- `src/petal/core/builders/` - Builder pattern implementation
- `src/petal/core/config/` - Configuration management
- `src/petal/core/steps/` - Step strategy implementations
- `src/petal/core/plugins/` - Plugin system for extensibility

### Key Design Patterns
- **Strategy Pattern**: Step management via StepStrategy classes
- **Builder Pattern**: Agent construction via AgentBuilder
- **Configuration Object**: Pydantic-based configuration management
- **Chain of Responsibility**: Step configuration handlers

---

## Quick Commands Reference

### Stage All Changes
```bash
uv run git add .
```

### Check Status
```bash
uv run git status
```

### Commit Changes
```bash
uv run git commit -m "descriptive commit message"
```

### Push to Remote
```bash
uv run git push origin main
```

### Pull Latest Changes
```bash
uv run git pull origin main
```

## Complete Workflow

### 1. Check Current Status
```bash
uv run git status
```
This shows:
- Current branch
- Staged changes
- Unstaged changes
- Untracked files

### 2. Stage Changes
```bash
# Stage all changes
uv run git add .

# Or stage specific files
uv run git add filename.py

# Or stage specific directories
uv run git add src/petal/

# Stage specific modules
uv run git add src/petal/core/builders/
uv run git add src/petal/core/config/
uv run git add src/petal/core/steps/
```

### 3. Verify Staged Changes
```bash
uv run git status
uv run git diff --cached
```

### 4. Commit Changes
```bash
# Simple commit
uv run git commit -m "Add MCP tool factory with namespace support"

# Or with more detail
uv run git commit -m "feat: Add MCP tool factory with automatic namespace support

- Add ToolFactory.add_mcp() method with optional resolver
- Implement automatic 'mcp:' prefix for MCP tools
- Add comprehensive test coverage for MCP integration
- Update config to use server key as MCP server name"
```

### 5. Push to Remote
```bash
# Push to current branch
uv run git push

# Or specify remote and branch
uv run git push origin main
```

### 6. Verify Push
```bash
# Check remote status
uv run git remote -v

# Check branch tracking
uv run git branch -vv
```

## Feature Branch Workflow

### Create and Switch to Feature Branch
```bash
uv run git checkout -b feature/mcp-tool-factory
```

### Work on Feature
```bash
# Make changes, then stage and commit
uv run git add .
uv run git commit -m "feat: Add MCP tool factory"
```

### Push Feature Branch
```bash
uv run git push -u origin feature/mcp-tool-factory
```

### Merge to Main (after review)
```bash
uv run git checkout main
uv run git pull origin main
uv run git merge feature/mcp-tool-factory
uv run git push origin main
```

## Troubleshooting

### Fix Corrupted Coverage Files
If pre-commit hooks fail with coverage errors, clean up corrupted coverage data:
```bash
rm -f .coverage*
uv run git add .
uv run git commit --no-verify -m "your commit message"
```

### Undo Last Commit (keep changes)
```bash
uv run git reset --soft HEAD~1
```

### Undo Last Commit (discard changes)
```bash
uv run git reset --hard HEAD~1
```

### Check Commit History
```bash
uv run git log --oneline -10
```

### Check Remote Branches
```bash
uv run git branch -r
```

## Pre-commit Checklist

Before committing, ensure:
1. All tests pass: `uv run make test`
2. Code is formatted: `uv run make pristine`
3. Type checking passes: `uv run make mypy`
4. Coverage is maintained: `uv run make coverage-check`

## Common Patterns

### Atomic Commits
```bash
# Stage related changes together
uv run git add src/petal/core/tool_factory.py
uv run git commit -m "feat: Add MCP tool registration"

uv run git add tests/petal/test_tool_factory.py
uv run git commit -m "test: Add MCP tool factory tests"
```

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Integration with Makefile

Use the Makefile targets for quality checks:

```bash
# Run all checks
uv run make checkit

# Format and lint
uv run make pristine

# Run tests
uv run make test

# Check coverage
uv run make coverage-check
```

## Architecture-Aware Development

When working with the new architecture:

### Adding New Step Types
1. Create new strategy in `src/petal/core/steps/`
2. Register in `StepRegistry`
3. Add tests in `tests/petal/test_steps_*.py`

### Modifying Configuration
1. Update models in `src/petal/core/config/`
2. Update validation in `AgentConfig`
3. Add tests in `tests/petal/test_config_*.py`

### Extending Builders
1. Add methods to `AgentBuilder`
2. Update `AgentBuilderDirector` if needed
3. Add tests in `tests/petal/test_builders_*.py`
