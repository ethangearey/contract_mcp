import asyncio
import sys
import os
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class MCPClient:
    def __init__(self, use_local_model: bool = False):
        """Initialize MCP Client

        Args:
            use_local_model: If True, use local model instead of Claude API
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.use_local_model = use_local_model

        if not use_local_model:
            self.anthropic = Anthropic()
        else:
            # For local model - you can modify this to use your preferred local API
            # This example assumes OpenAI-compatible API (like Ollama with OpenAI compatibility)
            try:
                from openai import AsyncOpenAI
                self.local_client = AsyncOpenAI(
                    base_url="http://localhost:11434/v1",  # Ollama default
                    api_key="ollama"  # Ollama doesn't need real API key
                )
            except ImportError:
                print("Warning: OpenAI library not installed. Install with: pip install openai")
                self.local_client = None

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        if not os.path.exists(server_script_path):
            raise FileNotFoundError(f"Server script not found: {server_script_path}")

        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')

        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print(f"\nConnected to server with tools: {[tool.name for tool in tools]}")
        return tools

    async def process_query_claude(self, query: str) -> str:
        """Process query using Claude API"""
        # Get available tools
        response = await self.session.list_tools()
        available_tools = []

        for tool in response.tools:
            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            available_tools.append(tool_def)

        messages = [{"role": "user", "content": query}]

        # Call Claude with tools
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        result_parts = []

        # Process Claude's response
        for content in response.content:
            if content.type == 'text':
                result_parts.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                print(f"🔧 Calling tool: {tool_name} with args: {tool_args}")

                # Execute the tool
                try:
                    tool_result = await self.session.call_tool(tool_name, tool_args)

                    # Extract text content from tool result
                    tool_output = ""
                    if hasattr(tool_result, 'content') and tool_result.content:
                        for content_item in tool_result.content:
                            if hasattr(content_item, 'text'):
                                tool_output += content_item.text

                    # Continue conversation with tool results
                    messages.append({
                        "role": "assistant",
                        "content": [content]  # Include the tool_use
                    })
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": content.id, "content": tool_output}]
                    })

                    # Get Claude's final response
                    final_response = self.anthropic.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1000,
                        messages=messages
                    )

                    for final_content in final_response.content:
                        if final_content.type == 'text':
                            result_parts.append(final_content.text)

                except Exception as e:
                    result_parts.append(f"❌ Tool execution failed: {str(e)}")

        return "\n".join(result_parts)

    async def process_query_local(self, query: str, model_name: str = "llama3.2") -> str:
        """Process query using local model"""
        if not self.local_client:
            return "❌ Local model client not available. Please install openai: pip install openai"

        # Get available tools
        response = await self.session.list_tools()
        available_tools = [tool.name for tool in response.tools]

        # Simple approach: ask the local model if it needs to use tools
        system_prompt = f"""You are a helpful assistant with access to these tools: {', '.join(available_tools)}.

If the user's query requires web search, respond with exactly: USE_TOOL:brave_search:{{query:"search terms", count:5}}

Otherwise, respond normally to the user's query."""

        try:
            response = await self.local_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=1000,
                temperature=0.7
            )

            response_text = response.choices[0].message.content

            # Check if model wants to use a tool
            if response_text.startswith("USE_TOOL:brave_search:"):
                tool_args_str = response_text.replace("USE_TOOL:brave_search:", "")
                try:
                    import json
                    tool_args = json.loads(tool_args_str)

                    print(f"🔧 Local model requesting tool: brave_search with args: {tool_args}")

                    # Execute the tool
                    tool_result = await self.session.call_tool("brave_search", tool_args)

                    # Extract tool output
                    tool_output = ""
                    if hasattr(tool_result, 'content') and tool_result.content:
                        for content_item in tool_result.content:
                            if hasattr(content_item, 'text'):
                                tool_output += content_item.text

                    # Get final response from local model with search results
                    final_response = await self.local_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system",
                             "content": "You are a helpful assistant. Use the search results to answer the user's question."},
                            {"role": "user", "content": query},
                            {"role": "assistant", "content": f"I found these search results: {tool_output}"},
                            {"role": "user", "content": "Please summarize and answer based on these results."}
                        ],
                        max_tokens=1000,
                        temperature=0.7
                    )

                    return final_response.choices[0].message.content

                except Exception as e:
                    return f"❌ Tool execution failed: {str(e)}"

            return response_text

        except Exception as e:
            return f"❌ Local model error: {str(e)}"

    async def process_query(self, query: str, local_model: str = "llama3.2") -> str:
        """Process query using selected model"""
        if self.use_local_model:
            return await self.process_query_local(query, local_model)
        else:
            return await self.process_query_claude(query)

    async def chat_loop(self, local_model: str = "llama3.2"):
        """Run interactive chat loop"""
        model_type = "Local Model" if self.use_local_model else "Claude API"
        print(f"\n🤖 MCP Client Started! Using: {model_type}")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\n💬 Query: ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    break

                if not query:
                    continue

                print("🔄 Processing...")
                response = await self.process_query(query, local_model)
                print(f"\n📝 Response:\n{response}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <path_to_server_script> [--local] [--model model_name]")
        print("  --local: Use local model instead of Claude API")
        print("  --model: Specify local model name (default: llama3.2)")
        sys.exit(1)

    server_script = sys.argv[1]
    use_local = "--local" in sys.argv

    # Get model name if specified
    local_model = "llama3.2"
    if "--model" in sys.argv:
        model_idx = sys.argv.index("--model")
        if model_idx + 1 < len(sys.argv):
            local_model = sys.argv[model_idx + 1]

    client = MCPClient(use_local_model=use_local)

    try:
        await client.connect_to_server(server_script)
        await client.chat_loop(local_model)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())