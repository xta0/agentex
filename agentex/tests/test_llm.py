"""LLM adapter tests with the provider call mocked out."""

import json
import unittest
from unittest.mock import AsyncMock, patch

from litellm import ModelResponse

from agentex.llm import LlmClient, LLMRequest
from agentex.tool import tool
from agentex.types import Message, ToolCall, ToolResult

USAGE = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}


class LlmClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_forwards_configuration_and_parses_text_and_usage(self):
        client = LlmClient("test-model", temperature=0.2, timeout=30)
        request = LLMRequest(
            instructions=["Be concise.", "Use plain language."],
            contents=[
                Message(role="user", content="Hi"),
                Message(role="assistant", content="Hello"),
                Message(role="user", content="What is 2 + 2?"),
            ],
        )
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "4"}}],
            usage=USAGE,
        )
        with patch(
            "agentex.llm.acompletion", new_callable=AsyncMock, return_value=response
        ) as mock_completion:
            result = await client.generate(request)

        mock_completion.assert_awaited_once_with(
            model="test-model",
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "system", "content": "Use plain language."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            tools=None,
            temperature=0.2,
            timeout=30,
        )
        self.assertEqual(result.contents, [Message(role="assistant", content="4")])
        self.assertEqual(
            result.usage_metadata, {"input_tokens": 12, "output_tokens": 3}
        )
        self.assertIsNone(result.error_message)

    async def test_generate_forwards_tool_definitions_and_choice(self):
        @tool
        def add(first: int, second: int) -> int:
            return first + second

        with patch(
            "agentex.llm.acompletion",
            new_callable=AsyncMock,
            return_value=ModelResponse(usage=USAGE),
        ) as mock_completion:
            result = await LlmClient("test-model").generate(
                LLMRequest(tools=[add], tool_choice="required")
            )

        self.assertIsNone(result.error_message)
        self.assertEqual(
            mock_completion.call_args.kwargs["tools"], [add.tool_definition]
        )
        self.assertEqual(mock_completion.call_args.kwargs["tool_choice"], "required")

    async def test_multiple_tool_calls_round_trip_with_ids_and_json_arguments(self):
        response = ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "I will check both.",
                        "tool_calls": [
                            {
                                "id": "call_search",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "café", "limit": 2}',
                                },
                            },
                            {
                                "id": "call_ping",
                                "type": "function",
                                "function": {"name": "ping", "arguments": "{}"},
                            },
                        ],
                    }
                }
            ],
            usage=USAGE,
        )
        client = LlmClient("test-model")
        with patch(
            "agentex.llm.acompletion", new_callable=AsyncMock, return_value=response
        ) as mock_completion:
            result = await client.generate(LLMRequest())
            self.assertIsNone(result.error_message)
            self.assertEqual(
                result.contents,
                [
                    Message(role="assistant", content="I will check both."),
                    ToolCall(
                        tool_call_id="call_search",
                        name="search",
                        arguments={"query": "café", "limit": 2},
                    ),
                    ToolCall(tool_call_id="call_ping", name="ping", arguments={}),
                ],
            )
            second_result = await client.generate(
                LLMRequest(
                    contents=[
                        *result.contents,
                        ToolResult(
                            tool_call_id="call_search",
                            name="search",
                            status="success",
                            content=["Found two results"],
                        ),
                        ToolResult(
                            tool_call_id="call_ping",
                            name="ping",
                            status="success",
                            content=[],
                        ),
                    ]
                )
            )

        self.assertIsNone(second_result.error_message)
        messages = mock_completion.call_args.kwargs["messages"]
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["content"], "I will check both.")
        calls = messages[0]["tool_calls"]
        self.assertEqual([call["id"] for call in calls], ["call_search", "call_ping"])
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"query": "café", "limit": 2},
        )
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {})
        self.assertEqual(
            messages[1:],
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_search",
                    "content": "Found two results",
                },
                {"role": "tool", "tool_call_id": "call_ping", "content": ""},
            ],
        )

    async def test_tool_call_without_text_creates_assistant_message(self):
        with patch(
            "agentex.llm.acompletion",
            new_callable=AsyncMock,
            return_value=ModelResponse(usage=USAGE),
        ) as mock_completion:
            result = await LlmClient("test-model").generate(
                LLMRequest(
                    contents=[
                        ToolCall(tool_call_id="call_1", name="ping", arguments={})
                    ]
                )
            )

        self.assertIsNone(result.error_message)
        self.assertEqual(
            mock_completion.call_args.kwargs["messages"],
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "ping", "arguments": "{}"},
                        }
                    ],
                }
            ],
        )

    async def test_malformed_tool_arguments_are_reported_as_response_error(self):
        response = ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "ping", "arguments": "not JSON"},
                            }
                        ],
                    }
                }
            ],
            usage=USAGE,
        )
        with patch(
            "agentex.llm.acompletion", new_callable=AsyncMock, return_value=response
        ):
            result = await LlmClient("test-model").generate(LLMRequest())

        self.assertIn("Expecting value", result.error_message)
        self.assertEqual(result.contents, [])

    async def test_provider_failure_is_returned_without_partial_contents(self):
        with patch(
            "agentex.llm.acompletion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Provider unavailable"),
        ):
            result = await LlmClient("test-model").generate(LLMRequest())

        self.assertEqual(result.error_message, "Provider unavailable")
        self.assertEqual(result.contents, [])
        self.assertEqual(result.usage_metadata, {})
