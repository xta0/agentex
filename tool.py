import asyncio
import json
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel
from litellm import completion

load_dotenv()


calculator_tool_definition = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {
                    "type": "string",
                    "description": "Arithmetic operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "first_number": {
                    "type": "number",
                    "description": "First number for the calculation"
                },
                "second_number": {
                    "type": "number",
                    "description": "Second number for the calculation"
                }
            },
            "required": ["operator", "first_number", "second_number"],
        }
    }
}

def calculator(operator: str, first_number: float, second_number: float):
    print(f"Calculating: {first_number} {operator} {second_number}")
    if operator == 'add':
        return first_number + second_number
    elif operator == 'subtract':
        return first_number - second_number
    elif operator == 'multiply':
        return first_number * second_number
    elif operator == 'divide':
        if second_number == 0:
            raise ValueError("Cannot divide by zero")
        return first_number / second_number
    else:
        raise ValueError(f"Unsupported operator: {operator}")




tools = [calculator_tool_definition]

response_without_tool = completion(
    model='gpt-5.4-mini',
    messages=[{"role": "user", "content": "What is the capital of South Korea?"}],
    tools=tools,
)
print("# Capital question doesn't need a tool call")
print(response_without_tool.choices[0].message.content)
print(response_without_tool.choices[0].message.tool_calls)

messages = [{"role": "user", "content": "What is 1234 x 5678?"}]
response_with_tool = completion(
    model='gpt-5.4-mini',
    messages=messages,
    tools=tools,
)
print("\n# Multiplication question needs a tool call")
print(response_with_tool.choices[0].message.content)
print(response_with_tool.choices[0].message.tool_calls)

ai_message = response_with_tool.choices[0].message
messages.append({
    "role": "assistant",
    "content": ai_message.content,
    "tool_calls": ai_message.tool_calls
})

if ai_message.tool_calls:
    for tool_call in ai_message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        if tool_name == "calculator":
            result = calculator(**tool_args)
            print(f"\n# Tool call result: {result}")
            messages.append({                    # C: append the tool result
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

final_response = completion(
    model='gpt-5.4-mini',
    messages=messages,
)
print("Messages:", messages)
print("Final Answer:", final_response.choices[0].message.content)


