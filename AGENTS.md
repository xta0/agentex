# Repository guidance

## Scope and project status

These instructions apply throughout this repository unless a more specific
`AGENTS.md` overrides them.

`agentex` is an early Python project for building an AI agent from scratch. Keep
changes small, readable, and consistent with the existing modules. `Agent` and
`LlmClient` are configuration scaffolds; do not assume a model-call method or an
agent execution loop exists. Describe current behavior accurately in docs and
examples.

## Repository map

- `agentex/agent.py`: agent configuration and the future execution loop.
- `agentex/llm.py`: model name and client options.
- `agentex/context.py`: Pydantic content/event models and `ExecutionContext` state.
- `agentex/tool.py`: tool interfaces, function wrappers, and the `@tool` decorator.
- `agentex/utils.py`: function-to-schema and tool-definition helpers.
- `agentex/mcp_tools.py`: MCP discovery, session handling, and tool wrappers.
- `agentex/main.py`: Tavily MCP discovery demo, not a full agent runner.
- `misc/`: exploratory model, search, calculator, MCP, and GAIA scripts.

Read the relevant implementation before editing. Experiments in `misc/` may lag
behind the core interfaces and are not authoritative API examples.

## Environment and commands

Use Python 3.12+ and `uv`. Dependencies live in `pyproject.toml`, with resolved
versions in `uv.lock`.

```sh
uv sync --all-groups --all-extras
make dev
make format
make lint
```

`make dev` syncs dependencies and opens an interactive Bash shell in `.venv`;
it does not start an application. Use `uv run` for noninteractive commands.
Run Python entry points as modules from the repository root, such as
`uv run python -m agentex.main`, so package imports work.

## Implementation conventions

- Follow Ruff formatting and the surrounding Python style. Use type annotations
  for new public interfaces and concise docstrings where behavior needs explaining.
- Keep reusable functionality in `agentex/` and exploratory scripts in `misc/`.
- Preserve asynchronous interfaces: tool execution is awaited, and a tool call
  receives an `ExecutionContext` before its tool arguments. Keep blocking I/O out
  of new async paths.
- `@tool` returns a `FunctionTool`, not the original function. Read its interface
  before changing callers or schema generation.
- Keep tool definitions consistent with the existing `type: function` envelope
  and nested `function` name, description, and parameters fields.
- Treat Pydantic content models, event fields, and tool-call identifiers as shared
  contracts. Check producers and consumers when changing them.
- Keep MCP sessions and subprocesses inside their async context managers. Preserve
  cleanup on errors, and keep stdout free of diagnostic output in stdio MCP servers.
- Do not add dependencies casually. If one is necessary, update `pyproject.toml`
  and regenerate `uv.lock` with `uv`; do not hand-edit the lockfile.

## Validation

There is currently no automated test suite or test runner configured. For Python
changes, run the relevant offline checks, preferably scoped to the files changed:

```sh
uv run ruff format --check path/to/changed_file.py
uv run ruff check path/to/changed_file.py
```

Use `make lint` for a repository-wide check when appropriate. Report existing
unrelated findings separately instead of expanding the change to fix them.
For behavior changes, exercise the affected path locally and add focused tests
when they provide meaningful regression coverage. For documentation-only changes,
verify paths, commands, links, and claims against the source.

Avoid live service calls as routine validation. The example scripts can launch
subprocesses, download packages or datasets, and consume paid API usage. Prefer
local functions, fakes, or mocks unless live integration testing is in scope.
State which checks ran and which integrations remain unverified.

## Secrets and change hygiene

- Load credentials from the environment or the ignored `.env` file. Never print
  real credentials or add them to source, documentation, logs, or commits.
- Do not commit `.env`, `.venv`, caches, downloaded datasets, or generated outputs.
- Inspect the working tree before editing, preserve existing user changes, and
  avoid unrelated cleanup or repository-wide formatting.
- Update `README.md` when setup, entry points, or supported behavior changes.
