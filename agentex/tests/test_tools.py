"""Function tool execution and generated schema contracts."""

import unittest

from agentex.context import ExecutionContext
from agentex.tool import FunctionTool, tool


class FunctionToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_decorator_wraps_sync_function_and_preserves_defaults(self):
        @tool
        def add(first: int, second: int = 3) -> int:
            """Add two integers."""
            return first + second

        self.assertIsInstance(add, FunctionTool)
        self.assertEqual(await add(ExecutionContext(), first=2), 5)
        self.assertEqual(
            add.tool_definition,
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "first": {"type": "integer"},
                            "second": {"type": "integer"},
                        },
                        "required": ["first"],
                    },
                },
            },
        )

    async def test_async_function_receives_execution_context(self):
        @tool
        async def remember(context: ExecutionContext, value: str) -> str:
            context.state["remembered"] = value
            return context.execution_id

        context = ExecutionContext()
        result = await remember(context, value="hello")
        self.assertEqual(result, context.execution_id)
        self.assertEqual(context.state, {"remembered": "hello"})

    async def test_sync_function_returning_awaitable_is_awaited(self):
        async def double(value: int) -> int:
            return value * 2

        @tool
        def deferred_double(value: int):
            return double(value)

        self.assertEqual(await deferred_double(ExecutionContext(), value=3), 6)

    async def test_decorator_accepts_custom_name_and_description(self):
        @tool(name="sum_numbers", description="Sum the supplied integers.")
        def add(first: int, second: int) -> int:
            return first + second

        self.assertEqual(add.name, "sum_numbers")
        self.assertEqual(add.description, "Sum the supplied integers.")
        self.assertEqual(add.tool_definition["function"]["name"], "sum_numbers")
        self.assertEqual(await add(ExecutionContext(), first=2, second=4), 6)

    async def test_explicit_definition_supports_complex_parameters(self):
        definition = {
            "type": "function",
            "function": {
                "name": "total",
                "description": "Add a list of integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "array", "items": {"type": "integer"}}
                    },
                    "required": ["values"],
                },
            },
        }

        def total(values: list[int]) -> int:
            return sum(values)

        wrapped = FunctionTool(total, tool_definition=definition)
        self.assertEqual(wrapped.tool_definition, definition)
        self.assertEqual(await wrapped(ExecutionContext(), values=[1, 2, 3]), 6)

    async def test_execution_errors_propagate_to_caller(self):
        @tool
        def divide(first: int, second: int) -> float:
            return first / second

        with self.assertRaises(ZeroDivisionError):
            await divide(ExecutionContext(), first=1, second=0)


class ToolSchemaTests(unittest.TestCase):
    def test_supported_builtin_annotations_and_keyword_only_defaults(self):
        @tool
        def options(
            text: str,
            count: int,
            ratio: float,
            enabled: bool,
            values: list,
            metadata: dict,
            missing: None,
            *,
            label: str = "default",
        ) -> None:
            pass

        parameters = options.tool_definition["function"]["parameters"]
        self.assertEqual(
            parameters["properties"],
            {
                "text": {"type": "string"},
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "enabled": {"type": "boolean"},
                "values": {"type": "array"},
                "metadata": {"type": "object"},
                "missing": {"type": "null"},
                "label": {"type": "string"},
            },
        )
        self.assertEqual(
            parameters["required"],
            ["text", "count", "ratio", "enabled", "values", "metadata", "missing"],
        )
