import inspect


def function_to_input_schema(func) -> dict:
    """Convert a function's signature to an input schema dictionary."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
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
