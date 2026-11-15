import inspect
from agentex.types import Message
from agentex.context import ToolCall, ToolResult


def function_to_input_schema(func) -> dict:
    """Convert a function's signature to an input schema dictionary."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        None: "null",
        type(None): "null",
    }
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Could not retrieve signature for function {func.__name__}: {e}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(
                param.annotation, "string"
            )  # Default to string if type not found
        except Exception as e:
            raise ValueError(
                f"Could not determine type for parameter {param.name}: {e}"
            )
        parameters[param.name] = {"type": param_type}

    # Parameters without default values are marked as required
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default is inspect.Parameter.empty
    ]

    return {
        "type": "object",
        "properties": parameters,
        "required": required,
    }


def format_tool_definition(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def funtion_to_tool_definition(func) -> dict:
    """Convert a function to a tool definition dictionary."""
    parameters = function_to_input_schema(func)
    return format_tool_definition(
        name=func.__name__,
        description=func.__doc__ or "No description provided.",
        parameters=parameters,
    )


def display_trace(context) -> None:
    """Print a readable trace of events recorded in an ExecutionContext."""
    for index, event in enumerate(context.events, start=1):
        print(f"\n[{index}] {event.author}")
        for item in event.content:
            if isinstance(item, Message):
                print(f"  {item.role}: {item.content}")
            elif isinstance(item, ToolCall):
                print(f"  tool call: {item.name}({item.arguments})")
            elif isinstance(item, ToolResult):
                preview = str(item.content[0]) if item.content else ""
                print(f"  tool result: {item.name} -> {preview[:500]}")
