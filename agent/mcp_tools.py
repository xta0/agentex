from .tool import BaseTool, FunctionTool
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from dotenv import load_dotenv

load_dotenv()


async def load_mcp_tools(connection: dict) -> list[BaseTool]:
    tools = []
    async with stdio_client(StdioServerParameters(**connection)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            provided_mcp_tools = await session.list_tools()
            for mcp_tool in provided_mcp_tools.tools:
                func_tool = _create_mcp_tool(mcp_tool, connection)
                tools.append(func_tool)
    return tools


def _create_mcp_tool(mcp_tool, connection: dict) -> FunctionTool:
    async def call_mcp(**kwargs):
        async with stdio_client(StdioServerParameters(**connection)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(mcp_tool.name, kwargs)
                return _extract_text_content(result)

    tool_definition = {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.input_schema,
        },
    }
    return FunctionTool(
        func=call_mcp,
        name=mcp_tool.name,
        description=mcp_tool.description,
        tool_definition=tool_definition,
    )


def _extract_text_content(result) -> str:
    """Extract plain text from an MCP CallToolResult."""
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)
