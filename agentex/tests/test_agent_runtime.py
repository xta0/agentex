"""Offline regression tests for agent tool calls and structured output."""

import asyncio
import json
import unittest
from typing import Literal
from unittest.mock import AsyncMock, patch

from litellm import ModelResponse
from pydantic import BaseModel

from agentex.agent import Agent
from agentex.context import Event, ExecutionContext, Message, ToolCall, ToolResult
from agentex.llm import LlmClient
from agentex.tool import tool
from agentex.types import ToolCall as SharedToolCall
from agentex.types import ToolResult as SharedToolResult


class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: list[str]


ANSWER = {
    "sentiment": "positive",
    "confidence": 0.92,
    "key_phrases": ["exceeded expectations", "recommend"],
}


def completion(
    name: str | None = None,
    arguments: dict | None = None,
    text: str | None = None,
) -> ModelResponse:
    return ModelResponse(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ]
                    if name
                    else None,
                }
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


class AgentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_output_uses_model_schema_and_returns_model(self):
        tools = []
        agent = Agent(
            model=LlmClient("unused"), tools=tools, output_type=SentimentAnalysis
        )
        context = ExecutionContext()
        with patch(
            "agentex.llm.acompletion",
            new_callable=AsyncMock,
            return_value=completion("final_answer", ANSWER),
        ) as mock_completion:
            result = await agent.run("This product exceeded my expectations!", context)

        self.assertIsInstance(result.output, SentimentAnalysis)
        self.assertEqual(result.output.model_dump(), ANSWER)
        self.assertIs(result.context, context)
        self.assertEqual(context.current_step, 1)
        self.assertEqual(tools, [])
        request = mock_completion.call_args.kwargs
        self.assertEqual(request["tool_choice"], "required")
        self.assertEqual(
            request["tools"][0]["function"]["parameters"],
            SentimentAnalysis.model_json_schema(),
        )
        call = context.events[1].content[0]
        tool_result = context.events[2].content[0]
        self.assertIs(ToolCall, SharedToolCall)
        self.assertIs(ToolResult, SharedToolResult)
        self.assertIsInstance(call, ToolCall)
        self.assertIsInstance(tool_result, ToolResult)
        self.assertEqual(tool_result.tool_call_id, call.tool_call_id)
        self.assertEqual(tool_result.type, "tool_result")
        self.assertIs(tool_result.content[0], result.output)

    async def test_intermediate_tool_is_awaited_and_history_is_serialized(self):
        @tool
        async def add(context: ExecutionContext, first: int, second: int) -> int:
            context.state["added"] = True
            return first + second

        agent = Agent(
            model=LlmClient("unused"), tools=[add], output_type=SentimentAnalysis
        )
        responses = [
            completion("add", {"first": 2, "second": 3}),
            completion("final_answer", ANSWER),
        ]
        with patch(
            "agentex.llm.acompletion", new_callable=AsyncMock, side_effect=responses
        ) as mock_completion:
            result = await agent.run("Add two numbers, then analyze the sentiment.")

        self.assertTrue(result.context.state["added"])
        self.assertEqual(result.context.current_step, 2)
        messages = mock_completion.call_args_list[1].kwargs["messages"]
        call = messages[1]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "add")
        self.assertEqual(
            json.loads(call["function"]["arguments"]), {"first": 2, "second": 3}
        )
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], call["id"])
        self.assertEqual(messages[2]["content"], "5")

    async def test_invalid_structured_output_can_be_corrected(self):
        agent = Agent(model=LlmClient("unused"), output_type=SentimentAnalysis)
        responses = [
            completion("final_answer", {**ANSWER, "sentiment": "invalid"}),
            completion("final_answer", ANSWER),
        ]
        with patch(
            "agentex.llm.acompletion", new_callable=AsyncMock, side_effect=responses
        ) as mock_completion:
            result = await agent.run("Analyze this text.")

        self.assertEqual(result.output.sentiment, "positive")
        tool_error = result.context.events[2].content[0]
        self.assertEqual(tool_error.status, "error")
        self.assertIn("sentiment", tool_error.content[0])
        messages = mock_completion.call_args_list[1].kwargs["messages"]
        self.assertIn("sentiment", messages[-1]["content"])

    async def test_provider_errors_are_reported_immediately(self):
        agent = Agent(model=LlmClient("unused"), output_type=SentimentAnalysis)
        with (
            patch(
                "agentex.llm.acompletion",
                new_callable=AsyncMock,
                side_effect=ValueError("Provider request rejected"),
            ) as mock_completion,
            self.assertRaisesRegex(
                RuntimeError, "LLM request failed: Provider request rejected"
            ),
        ):
            await agent.run("Analyze this text.")
        mock_completion.assert_awaited_once()

    async def test_step_limit_raises_instead_of_returning_none(self):
        agent = Agent(
            model=LlmClient("unused"), output_type=SentimentAnalysis, max_steps=2
        )
        context = ExecutionContext()
        with (
            patch(
                "agentex.llm.acompletion",
                new_callable=AsyncMock,
                return_value=completion("final_answer", {}),
            ) as mock_completion,
            self.assertRaisesRegex(RuntimeError, "within 2 steps"),
        ):
            await agent.run("Analyze this text.", context)
        self.assertEqual(mock_completion.await_count, 2)
        self.assertEqual(context.events[-1].content[0].status, "error")

    async def test_text_does_not_finish_a_structured_output_run(self):
        agent = Agent(
            model=LlmClient("unused"), output_type=SentimentAnalysis, max_steps=2
        )
        with patch(
            "agentex.llm.acompletion",
            new_callable=AsyncMock,
            side_effect=[
                completion(text="The sentiment is positive."),
                completion("final_answer", ANSWER),
            ],
        ):
            result = await agent.run("Analyze this text.")
        self.assertIsInstance(result.output, SentimentAnalysis)
        self.assertEqual(result.context.current_step, 2)

    async def test_plain_text_output_still_works(self):
        agent = Agent(model=LlmClient("unused"))
        with patch(
            "agentex.llm.acompletion",
            new_callable=AsyncMock,
            return_value=completion(text="Hello!"),
        ):
            result = await agent.run("Hi!")
        self.assertEqual(result.output, "Hello!")

    async def test_empty_responses_exhaust_step_limit(self):
        agent = Agent(model=LlmClient("unused"), max_steps=2)
        with (
            patch(
                "agentex.llm.acompletion",
                new_callable=AsyncMock,
                return_value=completion(),
            ) as mock_completion,
            self.assertRaisesRegex(RuntimeError, "within 2 steps"),
        ):
            await agent.run("Hi!")
        self.assertEqual(mock_completion.await_count, 2)

    async def test_unknown_tool_raises_with_tool_name(self):
        agent = Agent(model=LlmClient("unused"))
        with (
            patch(
                "agentex.llm.acompletion",
                new_callable=AsyncMock,
                return_value=completion("missing_tool", {}),
            ),
            self.assertRaisesRegex(ValueError, "Tool 'missing_tool' not found"),
        ):
            await agent.run("Use a tool.")

    async def test_tool_exception_is_returned_to_model_for_recovery(self):
        @tool
        def divide(first: int, second: int) -> float:
            return first / second

        agent = Agent(model=LlmClient("unused"), tools=[divide])
        with patch(
            "agentex.llm.acompletion",
            new_callable=AsyncMock,
            side_effect=[
                completion("divide", {"first": 1, "second": 0}),
                completion(text="Division by zero is undefined."),
            ],
        ) as mock_completion:
            result = await agent.run("Divide 1 by 0.")

        self.assertEqual(result.output, "Division by zero is undefined.")
        error = result.context.events[2].content[0]
        self.assertEqual(error.status, "error")
        self.assertEqual(error.name, "divide")
        self.assertEqual(error.tool_call_id, "call_1")
        self.assertIn("division by zero", error.content[0])
        second_request = mock_completion.call_args_list[1].kwargs
        self.assertEqual(second_request["tool_choice"], "auto")
        self.assertIn("division by zero", second_request["messages"][-1]["content"])

    async def test_zero_step_budget_does_not_call_model(self):
        agent = Agent(model=LlmClient("unused"), max_steps=0)
        with (
            patch("agentex.llm.acompletion", new_callable=AsyncMock) as mock_completion,
            self.assertRaisesRegex(RuntimeError, "within 0 steps"),
        ):
            await agent.run("Hi!")
        mock_completion.assert_not_awaited()

    async def test_tool_cancellation_propagates(self):
        @tool
        async def cancelled() -> None:
            raise asyncio.CancelledError

        agent = Agent(model=LlmClient("unused"), tools=[cancelled])
        with (
            patch(
                "agentex.llm.acompletion",
                new_callable=AsyncMock,
                return_value=completion("cancelled", {}),
            ) as mock_completion,
            self.assertRaises(asyncio.CancelledError),
        ):
            await agent.run("Start the tool.")
        mock_completion.assert_awaited_once()

    async def test_nested_structured_output_is_validated(self):
        class Report(BaseModel):
            analysis: SentimentAnalysis
            note: str = ""

        agent = Agent(model=LlmClient("unused"), output_type=Report)
        with patch(
            "agentex.llm.acompletion",
            new_callable=AsyncMock,
            return_value=completion("final_answer", {"analysis": ANSWER}),
        ) as mock_completion:
            result = await agent.run("Analyze this text.")

        self.assertIsInstance(result.output, Report)
        self.assertIsInstance(result.output.analysis, SentimentAnalysis)
        self.assertEqual(result.output.analysis.model_dump(), ANSWER)
        self.assertEqual(result.output.note, "")
        self.assertEqual(
            mock_completion.call_args.kwargs["tools"][0]["function"]["parameters"],
            Report.model_json_schema(),
        )

    async def test_existing_context_history_and_state_are_preserved(self):
        context = ExecutionContext(state={"saved": "value"}, current_step=1)
        history = Event(
            execution_id=context.execution_id,
            author="user",
            content=[Message(role="user", content="Earlier question")],
        )
        context.add_event(history)
        agent = Agent(model=LlmClient("unused"), instructions="Be concise.")
        with patch(
            "agentex.llm.acompletion",
            new_callable=AsyncMock,
            return_value=completion(text="Answer"),
        ) as mock_completion:
            result = await agent.run("Follow-up question", context)

        self.assertIs(result.context, context)
        self.assertIs(context.events[0], history)
        self.assertEqual(context.state, {"saved": "value"})
        self.assertEqual(context.current_step, 2)
        self.assertEqual(
            mock_completion.call_args.kwargs["messages"],
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Earlier question"},
                {"role": "user", "content": "Follow-up question"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
