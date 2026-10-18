from abc import ABC, abstractmethod
import inspect
from typing import Any, Callable, Dict

from . import utils

from .context import ExecutionContext


class BaseTool(ABC):
    def __init__(
        self, name: str, description: str, tool_definition: Dict[str, Any] = None
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition or {}

    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """
        Execute the tool with the given arguments.
        """
        pass

    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        """
        Allow the tool to be called like a function.
        """
        return await self.execute(context, **kwargs)


class FunctionTool(BaseTool):
    """
    A tool that wraps a function.
    """

    def __init__(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None,
    ):
        self.func = func
        self.needs_context = "context" in inspect.signature(func).parameters
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()
        definition = tool_definition or self._generate_tool_definition()

        super().__init__(self.name, self.description, definition)

    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        if self.needs_context:
            kwargs["context"] = context
        result = self.func(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _generate_tool_definition(self) -> Dict[str, Any]:
        """
        Generate a tool definition based on the function's signature.
        """
        parameters = utils.function_to_input_schema(self.func)
        return utils.format_tool_definition(self.name, self.description, parameters)


def tool(func=None, *, name=None, description=None):
    """Decorator to create a FunctionTool from a function."""

    def decorator(f):
        return FunctionTool(
            func=f,
            name=name,
            description=description,
        )

    if func is not None:
        return decorator(func)
    return decorator
