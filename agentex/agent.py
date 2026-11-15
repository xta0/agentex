from attr import dataclass
from pydantic import BaseModel

from agentex.context import Event, ExecutionContext, Message
from agentex.llm import LlmClient, LLMRequest, LLMResponse
from agentex.tool import BaseTool, FunctionTool
from agentex.types import ToolCall, ToolResult
from agentex.utils import format_tool_definition


@dataclass
class AgentResult:
    """
    Result of an agent's execution.
    """

    output: str | BaseModel
    context: ExecutionContext | None = None


class Agent:
    """Tool-calling agent with ReAct loop."""

    def __init__(
        self,
        model: LlmClient,
        tools: list[BaseTool] | None = None,
        instructions: str = "",
        max_steps: int = 10,
        output_type: type[BaseModel] | None = None,
        name: str = "agent",
        description: str = "",
    ):
        self.model = model
        self.llm_client = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.description = description
        self.output_tool_name: str | None = None
        self.output_type = output_type
        self.tools = self._setup_tools(tools or [])

    def _setup_tools(self, tools: list[BaseTool]) -> list[BaseTool]:
        output_type = self.output_type
        if output_type is not None:

            def final_answer(**kwargs) -> BaseModel:
                return output_type.model_validate(kwargs)

            description = (
                "Return the final structured answer matching the required schema."
            )
            tools = [
                *tools,
                FunctionTool(
                    final_answer,
                    name="final_answer",
                    description=description,
                    tool_definition=format_tool_definition(
                        "final_answer", description, output_type.model_json_schema()
                    ),
                ),
            ]
            self.output_tool_name = "final_answer"
        return tools

    async def run(
        self, user_input: str, context: ExecutionContext | None = None
    ) -> AgentResult:
        if context is None:
            context = ExecutionContext()

        user_event = Event(
            execution_id=context.execution_id,
            author="user",
            content=[Message(role="user", content=user_input)],
        )

        context.add_event(user_event)

        while context.final_result is None and context.current_step < self.max_steps:
            await self.step(context)

            # check if the last event is a final response
            last_event = context.events[-1] if context.events else None
            if self._is_final_response(last_event):
                context.final_result = self._extract_final_result(last_event)
                break

        if context.final_result is None:
            raise RuntimeError(
                f"Agent did not produce a final answer within {self.max_steps} steps."
            )
        return AgentResult(output=context.final_result, context=context)

    async def step(self, context: ExecutionContext) -> None:
        # prepare LLm request
        llm_request = self._prepare_llm_request(context)

        llm_response = await self.think(llm_request)
        if llm_response.error_message is not None:
            raise RuntimeError(f"LLM request failed: {llm_response.error_message}")

        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=llm_response.contents,
        )

        context.add_event(response_event)

        tool_calls = [c for c in llm_response.contents if isinstance(c, ToolCall)]
        if tool_calls:
            tool_results = await self.act(context, tool_calls)
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=tool_results,
            )
            context.add_event(tool_event)

        context.increment_step()

    async def think(self, llm_request: LLMRequest) -> LLMResponse:
        return await self.llm_client.generate(llm_request)

    async def act(
        self, context: ExecutionContext, tool_calls: list[ToolCall]
    ) -> list[ToolResult]:
        tools_dict = {tool.name: tool for tool in self.tools}
        tool_results = []
        for call in tool_calls:
            if call.name not in tools_dict:
                raise ValueError(f"Tool '{call.name}' not found")
            tool = tools_dict[call.name]
            try:
                result = await tool(context, **call.arguments)
                tool_results.append(
                    ToolResult(
                        tool_call_id=call.tool_call_id,
                        name=call.name,
                        status="success",
                        content=[result],
                    )
                )
            except Exception as e:  # noqa: BLE001 - Return tool failures to the model for correction.
                tool_results.append(
                    ToolResult(
                        tool_call_id=call.tool_call_id,
                        name=call.name,
                        status="error",
                        content=[str(e)],
                    )
                )
        return tool_results

    def _prepare_llm_request(self, context: ExecutionContext) -> LLMRequest:
        # flatten events into content items
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)

        # determine tool choice strategy
        if self.output_tool_name:
            tool_choice = "required"
        elif self.tools:
            tool_choice = "auto"
        else:
            tool_choice = None

        return LLMRequest(
            instructions=[self.instructions] if self.instructions else [],
            contents=flat_contents,
            tools=self.tools,
            tool_choice=tool_choice if self.tools else None,
        )

    def _is_final_response(self, event: Event | None) -> bool:
        if event is None:
            return False
        if self.output_tool_name:
            for item in event.content:
                if (
                    isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"
                ):
                    return True
            return False

        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
        has_answer = any(
            isinstance(c, Message) and c.role == "assistant" for c in event.content
        )
        return has_answer and not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event | None) -> str | BaseModel | None:
        if event is None:
            return None
        if self.output_tool_name:
            for item in event.content:
                if (
                    isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"
                    and item.content
                ):
                    return item.content[0]

        # Original logic for free-text responses
        for c in event.content:
            if isinstance(c, Message) and c.role == "assistant":
                return c.content
        return None
