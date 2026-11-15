# agentex

A small Python project for learning how to build an AI agent from scratch.
The core is still under development and currently supports:

- Asynchronous model calls through LiteLLM.
- An agent loop that executes tools until a final answer or a step limit.
- Function tools created with `@tool`, including async functions.
- Structured output validated against a Pydantic model.
- MCP tool discovery and execution over stdio.
- Execution history and state, plus a helper for printing event traces.

The repository also includes a GAIA benchmark experiment and standalone examples
using LiteLLM and Tavily.

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for dependencies and commands.
- Credentials and model access for examples that call external services.
- Node.js and `npx` for the Tavily MCP server used by the GAIA experiment.
- Hugging Face dataset access and authentication when running GAIA.

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

The examples below use an OpenAI model. Other LiteLLM providers require the
corresponding model identifier and credentials. Local tool execution and the unit
tests do not require API keys.

`.env` is ignored by Git. Keep real keys out of source code, documentation, and
commits.

## Running an agent

Run Python code from the repository root so imports of `agentex` resolve. Save
this example as `example.py`, then run `uv run python example.py`. It makes live
model calls and requires `OPENAI_API_KEY` in the environment or `.env`.

```python
import asyncio

from dotenv import load_dotenv

from agentex import Agent
from agentex.llm import LlmClient
from agentex.tool import tool
from agentex.utils import display_trace

load_dotenv()


@tool
def add(first: int, second: int) -> int:
    """Add two integers."""
    return first + second


async def main() -> None:
    agent = Agent(
        model=LlmClient("gpt-5-mini"),
        tools=[add],
        instructions="Use the add tool for addition, then report the result.",
        max_steps=5,
    )
    result = await agent.run("What is 1234 + 5678?")
    print(result.output)
    display_trace(result.context)


if __name__ == "__main__":
    asyncio.run(main())
```

`Agent.run()` returns an `AgentResult` containing the final output and an
`ExecutionContext`. The context records conversation events, tool calls and
results, the current step, and a mutable `state` dictionary. `display_trace()`
prints the recorded events for inspection.

Each step makes one model request and executes any tool calls in the response.
The default limit is 10 steps. Tool execution errors are recorded and sent back
to the model so it can try again. An unknown tool name raises `ValueError`.
Model request or response parsing errors, and runs that exhaust their step
budget without a final answer, raise `RuntimeError`.

## Structured output

Pass a Pydantic model class as `output_type` to receive a validated model instance
in `result.output`:

```python
import asyncio
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from agentex import Agent
from agentex.llm import LlmClient

load_dotenv()


class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: list[str]


async def main() -> None:
    agent = Agent(
        model=LlmClient("gpt-5-mini"),
        instructions="Analyze the sentiment of the provided text.",
        output_type=SentimentAnalysis,
    )
    result = await agent.run("This product exceeded my expectations!")
    print(result.output.sentiment)
    print(result.output.confidence)
    print(result.output.key_phrases)


if __name__ == "__main__":
    asyncio.run(main())
```

The agent adds a `final_answer` tool using the model's JSON schema. Its arguments
are validated with Pydantic before the run completes. Invalid arguments become
tool errors and can be corrected on a later step. A plain text response does not
complete a run configured with `output_type`.

For direct model calls, use `LlmClient.generate()` with an `LLMRequest`. It returns
an `LLMResponse` containing content items, token usage metadata, and an
`error_message` when a request or parsing fails. The agent checks that error
field and raises it as a `RuntimeError`.

## Development and tests

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

`make test` runs Python's built-in `unittest` discovery in `agentex/tests/`.
After dependencies are installed, the tests run offline with mocked LLM calls
and MCP sessions. They require no API credentials or server subprocesses.

| Test module | Coverage |
| --- | --- |
| `test_agent_runtime.py` | Agent completion, structured output, validation retries, error handling, cancellation, step limits, and execution history. |
| `test_llm.py` | Request options, message conversion, tool-call identifiers and JSON arguments, usage metadata, and response errors. |
| `test_tools.py` | Sync and async tool execution, context injection, custom definitions, and input schemas. |
| `test_mcp_tools.py` | Discovery, wrapped calls, text extraction, and session cleanup on success and failure. |

Run a single module or test with:

```sh
uv run python -m unittest agentex.tests.test_agent_runtime
uv run python -m unittest agentex.tests.test_agent_runtime.AgentRuntimeTests.test_nested_structured_output_is_validated
```

Some experimental code may produce Ruff findings; review the output before
changing unrelated files.

## GAIA experiment

```sh
uv run python -m agentex.gaia
```

The current script loads the first three questions from the GAIA 2023 Level 1
validation split and evaluates them with `gpt-5` and `gpt-5-mini`, with up to three
concurrent solves. Each agent uses Tavily MCP tools and returns a structured
`GaiaOutput` with solvability, an answer, and a reason when unsolvable. The script
prints execution traces and results, comparing answers after trimming whitespace
and converting to lowercase.

This run requires OpenAI and Tavily credentials, access to the GAIA dataset on
Hugging Face, and Node.js with `npx`. It launches `tavily-mcp@latest`, which may be
downloaded, and makes live model and search calls. Only the question text is
passed to the agent; dataset attachments are not loaded into its context. Change
`MODELS` and the selected dataset range in `agentex/gaia.py` to adjust the run.

## Other experiments

Run these modules from the repository root. They are exploratory scripts and may
lag behind the core interfaces. Their external service integrations are not
covered by the unit suite; model and search calls may incur API charges.

| Command | Purpose | Requirements and current limits |
| --- | --- | --- |
| `uv run python -m misc.main` | Make concurrent LiteLLM calls with a semaphore and retries. | OpenAI key and access to the model selected in the script. |
| `uv run python -m misc.tool_calculator` | Demonstrate a model requesting a calculator tool and receiving its result. | OpenAI key and access to the model selected in the script. |
| `uv run python -m misc.client` | Discover tools from the local Tavily MCP server. | Tavily key; launches `misc/tavily_mcp_server.py` in a subprocess. |

`misc/tool_web_search.py` is an unfinished search-loop experiment. Its call sites
still treat decorated tools as plain synchronous functions, while `@tool` returns
an asynchronous `FunctionTool` that requires an execution context.

## Using a function tool locally

The tool abstraction can be used locally without API credentials:

```python
import asyncio

from agentex.context import ExecutionContext
from agentex.tool import tool


@tool
def add(first: int, second: int) -> int:
    """Add two integers."""
    return first + second


async def main() -> None:
    context = ExecutionContext()
    print(add.tool_definition)
    print(await add(context, first=2, second=3))  # 5


asyncio.run(main())
```

`@tool` returns a `FunctionTool`. Calls are awaited and receive an
`ExecutionContext` before the tool arguments. A wrapped function parameter named
`context` receives that context automatically. Synchronous functions run directly;
functions that perform blocking I/O should use an asynchronous implementation.

Generated input schemas support basic annotations such as `str`, `int`, `float`,
`bool`, `list`, `dict`, and `None`. They do not fully represent complex annotations
or exclude `context` from the schema. Supply an explicit `tool_definition` through
`FunctionTool` when the inferred schema is insufficient. Structured output uses
Pydantic's schema generation instead.

## Project layout

```text
agentex/
  agent.py       Tool-calling loop and structured output
  llm.py         Asynchronous LLM client using LiteLLM
  context.py     Events and execution state; re-exports shared content models
  tool.py        BaseTool, FunctionTool, and the @tool decorator
  utils.py       Input schemas, tool-definition helpers, and event trace display
  mcp_tools.py   MCP discovery and wrappers around remote tools
  types.py       Shared message, tool-call, and tool-result models
  gaia.py        GAIA evaluation using structured output and Tavily MCP tools
  tests/         Offline unit tests
misc/            Standalone experiments and a local Tavily MCP server
Makefile         Development shell, formatting, lint, and test commands
pyproject.toml   Python requirements and dependency groups
uv.lock          Locked dependency versions
```

See [AGENTS.md](AGENTS.md) for repository guidance for coding agents and
contributors.
