import asyncio
import os
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

from agent.utils import format_tool_definition

load_dotenv()

# server_params = StdioServerParameters(
#     command="npx",
#     args=["-y", "tavily-mcp@latest"],
#     env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
# )

# points to our own MCP server
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).with_name("tavily_mcp_server.py"))],
    env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")},
)


def mcp_tools_to_openai_format(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to OpenAI tool format."""
    return [
        format_tool_definition(
            name=tool.name,
            description=tool.description,
            parameters=tool.input_schema,
        )
        for tool in mcp_tools.tools
    ]


async def main():
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # list available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            openai_tools = mcp_tools_to_openai_format(tools_result)
            print("OpenAI tools format:")
            for tool in openai_tools:
                # print(f"- {tool['name']}: {tool['description']}")
                function = tool["function"]
                print(f"- {function['name']}: {function['description']}")

            # for tool in tools_result.tools:
            #     print(f"- {tool.name}: {tool.description}")

            # result = await session.call_tool(
            #     "tavily-search",
            #     arguments = { "query": "2025 FIBA Women's Basketball World Cup winner" }
            # )

            # print("Search result: ", result.content)


if __name__ == "__main__":
    asyncio.run(main())
