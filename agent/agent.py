from .llm import LlmClient
from .tool import BaseTool
from typing import List, Optional, Type


class Agent:
    """Tool-calling agent with ReAct loop."""

    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] | None = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
        description: str = "",
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.description = description
        self.output_tool_name: str | None = None
        self.tools = tools or []
