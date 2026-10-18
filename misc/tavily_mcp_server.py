import os
from tavily import TavilyClient
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

load_dotenv()

tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
mcp = MCPServer("custom-tavily-search")


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web using the Tavily client.

    Args:
        query (str): The search query.
        max_results (int, optional): The maximum number of results to return. Defaults to 5.

    Returns:
        str: The search results as a string.
    """
    try:
        response = tavily_client.search(query, max_results=max_results)
        results = response.get("results", [])
        return "\n\n".join(
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
            for r in results
        )
    except Exception as e:
        return f"Error during web search: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
