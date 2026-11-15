"""MCP discovery and tool lifecycle tests without subprocesses or services."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agentex.context import ExecutionContext
from agentex.mcp_tools import load_mcp_tools


class McpToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.connection = {"command": "unused", "args": ["--test"]}
        self.remote_tool = SimpleNamespace(
            name="search",
            description="Search documents.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        self.session = SimpleNamespace(
            initialize=AsyncMock(),
            list_tools=AsyncMock(
                return_value=SimpleNamespace(tools=[self.remote_tool])
            ),
            call_tool=AsyncMock(),
        )
        self.transport = MagicMock()
        self.transport.__aenter__.return_value = ("reader", "writer")
        self.session_manager = MagicMock()
        self.session_manager.__aenter__.return_value = self.session
        self.stdio_client = self.enterContext(
            patch("agentex.mcp_tools.stdio_client", return_value=self.transport)
        )
        self.client_session = self.enterContext(
            patch("agentex.mcp_tools.ClientSession", return_value=self.session_manager)
        )

    async def test_discovery_preserves_remote_definition_and_closes_contexts(self):
        tools = await load_mcp_tools(self.connection)

        self.assertEqual(len(tools), 1)
        self.assertEqual(
            tools[0].tool_definition,
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search documents.",
                    "parameters": self.remote_tool.input_schema,
                },
            },
        )
        self.assertEqual(self.stdio_client.call_args.args[0].command, "unused")
        self.assertEqual(self.stdio_client.call_args.args[0].args, ["--test"])
        self.client_session.assert_called_once_with("reader", "writer")
        self.session.initialize.assert_awaited_once()
        self.session.list_tools.assert_awaited_once()
        self.session.call_tool.assert_not_awaited()
        self.session_manager.__aexit__.assert_awaited_once_with(None, None, None)
        self.transport.__aexit__.assert_awaited_once_with(None, None, None)

    async def test_empty_server_returns_no_tools(self):
        self.session.list_tools.return_value = SimpleNamespace(tools=[])
        self.assertEqual(await load_mcp_tools(self.connection), [])
        self.session_manager.__aexit__.assert_awaited_once()
        self.transport.__aexit__.assert_awaited_once()

    async def test_discovery_failure_closes_both_contexts(self):
        self.session.list_tools.side_effect = RuntimeError("Discovery failed")
        with self.assertRaisesRegex(RuntimeError, "Discovery failed"):
            await load_mcp_tools(self.connection)

        self.assertIs(self.session_manager.__aexit__.call_args.args[0], RuntimeError)
        self.assertIs(self.transport.__aexit__.call_args.args[0], RuntimeError)

    async def test_initialization_failure_closes_both_contexts(self):
        self.session.initialize.side_effect = RuntimeError("Initialization failed")
        with self.assertRaisesRegex(RuntimeError, "Initialization failed"):
            await load_mcp_tools(self.connection)

        self.session.list_tools.assert_not_awaited()
        self.session_manager.__aexit__.assert_awaited_once()
        self.transport.__aexit__.assert_awaited_once()

    async def test_wrapped_call_reopens_session_and_extracts_only_text(self):
        tools = await load_mcp_tools(self.connection)
        self.session.call_tool.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(text="First result"),
                SimpleNamespace(type="image", data="unused"),
                SimpleNamespace(text=None),
                SimpleNamespace(text="Second result"),
            ]
        )

        result = await tools[0](ExecutionContext(), query="test query")

        self.assertEqual(result, "First result\nSecond result")
        self.session.call_tool.assert_awaited_once_with(
            "search", {"query": "test query"}
        )
        self.assertEqual(self.stdio_client.call_count, 2)
        self.assertEqual(self.session.initialize.await_count, 2)
        self.assertEqual(self.session_manager.__aexit__.await_count, 2)
        self.assertEqual(self.transport.__aexit__.await_count, 2)

    async def test_wrapped_call_handles_missing_or_empty_content(self):
        tools = await load_mcp_tools(self.connection)
        for response in (
            None,
            SimpleNamespace(),
            SimpleNamespace(content=None),
            SimpleNamespace(content=[]),
        ):
            with self.subTest(response=response):
                self.session.call_tool.return_value = response
                self.assertEqual(await tools[0](ExecutionContext(), query="test"), "")

    async def test_tool_failure_closes_session_and_transport(self):
        tools = await load_mcp_tools(self.connection)
        self.session.call_tool.side_effect = RuntimeError("Tool failed")
        with self.assertRaisesRegex(RuntimeError, "Tool failed"):
            await tools[0](ExecutionContext(), query="test")

        self.assertEqual(self.session_manager.__aexit__.await_count, 2)
        self.assertEqual(self.transport.__aexit__.await_count, 2)
        self.assertIs(self.session_manager.__aexit__.call_args.args[0], RuntimeError)
        self.assertIs(self.transport.__aexit__.call_args.args[0], RuntimeError)
