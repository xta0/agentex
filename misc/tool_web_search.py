import os
import json
from dotenv import load_dotenv
from tavily import TavilyClient
from agent.utils import funtion_to_tool_definition
from litellm import completion
from agent.tool import tool

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(
    query: str,
    max_results: int = 2,
    topic: str = "general",
    time_range: str | None = None,
    country: str | None = None,
) -> list:
    """Perform a web search using Tavily API and return the results."""
    try:
        response = tavily_client.search(
            query,
            max_results=max_results,
            topic=topic,
            time_range=time_range,
            country=country,
        )
        return response.get("results")
    except Exception as e:
        return f"Error during web search: {str(e)}"


def tool_execution(tool_box, tool_call):
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)
    result = tool_box[func_name](**func_args)
    return result


def simple_agent_loop(system_prompt, question):
    tools = [web_search]
    tool_box = {tool.__name__: tool for tool in tools}
    tool_defs = [funtion_to_tool_definition(tool) for tool in tools]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    while True:
        response = completion(
            model="gpt-5.4-mini",
            messages=messages,
            tools=tool_defs,
        )

        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                tool_result = tool_execution(tool_box, tool_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result),
                    }
                )
        else:
            return assistant_message.content


def main():
    system_prompt = """You are a helpful assistant.
    Use the search tool when you need current information.
    """
    result = simple_agent_loop(
        system_prompt, "Who won the 2026 FIBA Women's Basketball World Cup?"
    )
    print(result)


if __name__ == "__main__":
    main()
