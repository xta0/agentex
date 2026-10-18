import asyncio
import json
import os
from .mcp_tools import load_mcp_tools
from dotenv import load_dotenv
from .agent import Agent

load_dotenv()


async def main():
    connection = {
        "command": "npx",
        "args": ["-y", "tavily-mcp@latest"],
        "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")},
    }
    mcp_tools = await load_mcp_tools(connection)
    print(json.dumps([tool.tool_definition for tool in mcp_tools], indent=2))
    agent = Agent(
        model=None,  # Replace with an actual LLM client instance
        tools=mcp_tools,
        instructions="Use the tools to perform tasks.",
        max_steps=5,
        name="MCP Agent",
        description="An agent that uses MCP tools.",
    )


if __name__ == "__main__":
    asyncio.run(main())
