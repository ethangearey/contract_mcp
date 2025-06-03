#!/usr/bin/env python3
import os
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP # I think new MCP uses mcp.server.mcp
from dotenv import load_dotenv

# Initialize FastMCP server
mcp = FastMCP("brave-search")

# Replace with your actual Brave API key
load_dotenv()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")


@mcp.tool()
async def brave_search(query: str, count: int = 5) -> str:
    """Search the web using Brave Search API.

    Args:
        query: The search query
        count: Number of results to return (default: 5, max: 20)
    """
    if not query:
        return "Error: Query is required"

    if not BRAVE_API_KEY or BRAVE_API_KEY == "your_brave_api_key_here":
        return "Error: Brave API key not configured. Please set BRAVE_API_KEY environment variable."

    # Clamp count to valid range
    count = max(1, min(count, 20))

    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_API_KEY
            }
            params = {
                "q": query,
                "count": count
            }

            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=30.0
            )

            if response.status_code != 200:
                return f"Search failed with status {response.status_code}: {response.text}"

            data = response.json()
            results = data.get("web", {}).get("results", [])

            if not results:
                return "No search results found."

            # Format results
            formatted_results = []
            for i, result in enumerate(results[:count], 1):
                title = result.get("title", "No title")
                url = result.get("url", "No URL")
                description = result.get("description", "No description")
                formatted_results.append(f"{i}. **{title}**\n   URL: {url}\n   {description}\n")

            result_text = f"Search results for '{query}':\n\n" + "\n".join(formatted_results)
            return result_text

    except Exception as e:
        return f"Error performing search: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport='stdio')