from typing import Any, Literal, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import uuid


class Message(BaseModel):
    """
    A text message in the conversation.
    """

    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str


class ToolCall(BaseModel):
    """
    A tool call in the conversation.
    """

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    """
    A response to a tool call in the conversation.
    """

    type: Literal["tool_response"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list


ContentItem = Union[Message, ToolCall, ToolResult]


class Event(BaseModel):
    """
    An event in the conversation, which can be a message, tool call, or tool result.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str
    content: List[ContentItem]


@dataclass
class ExecutionContext:
    """
    Central storage for all execution state.
    """

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = field(default_factory=lambda: 0)
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None

    def add_event(self, event: Event):
        """
        Append an event to the execution history.
        """
        self.events.append(event)

    def increment_step(self):
        """
        Move to the next execution step.
        """
        self.current_step += 1
