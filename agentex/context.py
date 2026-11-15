import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agentex.types import ContentItem, Message, ToolCall, ToolResult

__all__ = [
    "ContentItem",
    "Event",
    "ExecutionContext",
    "Message",
    "ToolCall",
    "ToolResult",
]


class Event(BaseModel):
    """
    An event in the conversation, which can be a message, tool call, or tool result.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str
    content: list[ContentItem]


@dataclass
class ExecutionContext:
    """
    Central storage for all execution state.
    """

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: list[Event] = field(default_factory=list)
    current_step: int = field(default_factory=lambda: 0)
    state: dict[str, Any] = field(default_factory=dict)
    final_result: str | BaseModel | None = None

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
