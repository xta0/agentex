# agentex

A Python project for learning how to build an AI agent from scratch. It contains
tool abstractions, conversation state, MCP integration, and experiments with
LiteLLM, Tavily search, and the GAIA benchmark.

The core agent is still under development. `Agent.run()` executes a bounded
tool-calling loop, and `LlmClient` makes asynchronous model calls through LiteLLM.
With `output_type=YourPydanticModel`, the agent exposes that model's JSON schema
as a `final_answer` tool and returns a validated model instance in `result.output`.
Model errors and runs that reach `max_steps` without a final answer raise
`RuntimeError`.

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for dependencies and commands.
- Credentials for the external services used by the examples you choose to run.

## Setup

Run these commands from the repository root:

```sh
uv sync --all-groups --all-extras
```

Create a local `.env` file with the keys needed for your examples:

```dotenv
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

The LiteLLM examples currently select OpenAI models in their source files. Other
providers require a corresponding model identifier and credentials. The GAIA
experiment also requires access to the dataset through Hugging Face and local
authentication when required by that access.

`.env` is ignored by Git. Keep real keys out of source code, documentation, and
commits.

## Local development

```sh
make dev     # Sync dependencies and enter an interactive Bash shell in .venv
make format  # Format Python files with Ruff
make lint    # Check Python files with Ruff
make test    # Run the offline unit tests
```

Type `exit` to leave the development shell. You can also run commands directly
with `uv run` without activating a shell:

```sh
uv run ruff format --check .
uv run ruff check .
```

`make test` discovers all unit tests in `agentex/tests/`. The tests use Python's
built-in `unittest` runner and mock LLM calls and MCP sessions, so no credentials,
network access, or server subprocesses are needed. They cover the agent loop,
structured output, LLM message conversion, function tools, schemas, and MCP cleanup.
You can also run a single test module:

```sh
uv run python -m unittest agentex.tests.test_agent_runtime
```

Some experimental code may produce Ruff findings; review the output before
changing unrelated files.

## Examples

Run examples as modules from the repository root so imports of `agentex` resolve.
These are development examples, and their external service integrations have not
been verified by this documentation change. Service-backed examples need the
appropriate credentials and may incur API usage charges.

| Command | Purpose | Requirements and current limits |
| --- | --- | --- |
| `uv run python -m misc.main` | Make concurrent LiteLLM calls with a semaphore and retries. | OpenAI key and access to the model selected in the script. |
| `uv run python -m misc.tool_calculator` | Demonstrate a model requesting a calculator tool and receiving its result. | OpenAI key and access to the model selected in the script. |
| `uv run python -m misc.client` | Discover tools from the local Tavily MCP server. | Tavily key; launches `misc/tavily_mcp_server.py` in a subprocess. |
| `uv run python -m misc.gaia` | Evaluate the first 20 GAIA Level 1 validation questions against the configured models. | OpenAI key, model access, and Hugging Face dataset access; downloads data and makes multiple model calls. |

`misc/tool_web_search.py` is an unfinished search-loop experiment. Its call sites
still treat decorated tools as plain synchronous functions, while `@tool` returns
an asynchronous `FunctionTool` that requires an execution context.

## Using a function tool

The tool abstraction can be used locally without API credentials:

```python
import asyncio

from agentex.context import ExecutionContext
from agentex.tool import tool


@tool
def add(first: int, second: int) -> int:
    """Add two integers."""
    return first + second


async def main():
    context = ExecutionContext()
    print(add.tool_definition)
    print(await add(context, first=2, second=3))  # 5


asyncio.run(main())
```

`FunctionTool` wraps synchronous or asynchronous functions and awaits results
when needed. A function parameter named `context` receives the execution context
automatically. Schema generation currently handles a small set of basic Python
types; it does not fully represent complex annotations or exclude `context` from
the generated input schema. Supply an explicit definition through `FunctionTool`
when the inferred schema is insufficient.

## Project layout

```text
agentex/
  agent.py       Tool-calling loop and structured output
  llm.py         Asynchronous LLM client using LiteLLM
  context.py     Messages, tool calls, tool results, events, and execution state
  tool.py        BaseTool, FunctionTool, and the @tool decorator
  utils.py       Function input schemas and tool-definition helpers
  mcp_tools.py   MCP discovery and wrappers around remote tools
  types.py       Shared message, tool-call, and tool-result models
  tests/         Offline unit tests
misc/            Standalone experiments and a local Tavily MCP server
Makefile         Development shell, formatting, lint, and test commands
pyproject.toml   Python requirements and dependency groups
uv.lock          Locked dependency versions
```

See [AGENTS.md](AGENTS.md) for repository guidance for coding agents and
contributors.
