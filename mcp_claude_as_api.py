import asyncio
import os
import sys
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    use_local_model: bool = False
    local_model: str = "llama3.2"


class QueryResponse(BaseModel):
    response: str
    error: Optional[str] = None


class MCPClient:
    def __init__(self, use_local_model: bool = False):
        """Initialize MCP Client"""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.use_local_model = use_local_model
        self.is_connected = False

        if not use_local_model:
            self.anthropic = Anthropic()
        else:
            try:
                from openai import AsyncOpenAI
                self.local_client = AsyncOpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama"
                )
            except ImportError:
                print("Warning: OpenAI library not installed.")
                self.local_client = None

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        if self.is_connected:
            return

        if not os.path.exists(server_script_path):
            raise FileNotFoundError(f"Server script not found: {server_script_path}")

        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')

        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = sys.executable if is_python else "node"
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
        self.is_connected = True

        response = await self.session.list_tools()
        tools = response.tools
        print(f"Connected to server with tools: {[tool.name for tool in tools]}")
        return tools

    async def process_query_claude(self, query: str) -> str:
        """Process query using Claude API with support for multiple tool call rounds"""
        try:
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
            max_iterations = 5  # Prevent infinite loops
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                print(f"🔄 Claude interaction round {iteration}")

                # Get response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                # Process the response
                assistant_content = []
                tool_calls = []

                for content in response.content:
                    if content.type == 'text':
                        assistant_content.append({"type": "text", "text": content.text})
                    elif content.type == 'tool_use':
                        assistant_content.append({
                            "type": "tool_use",
                            "id": content.id,
                            "name": content.name,
                            "input": content.input
                        })
                        tool_calls.append(content)

                # Add assistant message
                messages.append({"role": "assistant", "content": assistant_content})

                # If no tool calls, we're done - return the text response
                if not tool_calls:
                    text_parts = [c["text"] for c in assistant_content if c["type"] == "text"]
                    final_response = "\n".join(text_parts)
                    print(f"✅ Final response (no more tools): {len(final_response)} characters")
                    return final_response

                # Execute each tool call
                print(f"🔧 Executing {len(tool_calls)} tool call(s)")
                for tool_call in tool_calls:
                    tool_name = tool_call.name
                    tool_args = tool_call.input
                    print(f"🔧 Calling tool: {tool_name} with args: {tool_args}")

                    try:
                        tool_result = await self.session.call_tool(tool_name, tool_args)
                        tool_output = ""

                        if hasattr(tool_result, 'content') and tool_result.content:
                            for content_item in tool_result.content:
                                if hasattr(content_item, 'text'):
                                    tool_output += content_item.text

                        print(f"✅ Tool result length: {len(tool_output)} characters")

                        # Add tool result to conversation
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": tool_output
                            }]
                        })

                    except Exception as e:
                        print(f"❌ Tool execution failed: {str(e)}")
                        # Add error result
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": f"Error: {str(e)}"
                            }]
                        })

            # If we hit max iterations, return what we have
            print(f"⚠️ Reached max iterations ({max_iterations})")
            return "Response truncated due to too many tool calls."

        except Exception as e:
            error_msg = f"Error in process_query_claude: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    async def process_query(self, query: str, local_model: str = "llama3.2") -> str:
        """Process query using selected model"""
        if self.use_local_model:
            return "Local model processing not implemented in this simplified version"
        else:
            return await self.process_query_claude(query)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
        self.is_connected = False


# Global client instances
claude_client: Optional[MCPClient] = None
local_client: Optional[MCPClient] = None
initialization_status = {"claude": False, "local": False, "error": None}


async def initialize_mcp_clients():
    """Initialize MCP clients in background"""
    global claude_client, local_client, initialization_status

    server_script = os.getenv("MCP_SERVER_SCRIPT")
    if not server_script:
        initialization_status["error"] = "MCP_SERVER_SCRIPT environment variable not set"
        return

    try:
        print("🔧 Initializing Claude client...")
        claude_client = MCPClient(use_local_model=False)
        await claude_client.connect_to_server(server_script)
        initialization_status["claude"] = True
        print("✅ Claude client connected")

        print("🔧 Initializing local client...")
        local_client = MCPClient(use_local_model=True)
        await local_client.connect_to_server(server_script)
        initialization_status["local"] = True
        print("✅ Local client connected")

    except Exception as e:
        error_msg = f"Error initializing MCP clients: {e}"
        print(f"❌ {error_msg}")
        initialization_status["error"] = error_msg


# Simplified lifespan that doesn't block
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    print("🚀 Starting API server...")

    # Start MCP client initialization in background (non-blocking)
    asyncio.create_task(initialize_mcp_clients())

    yield  # Application runs here

    # Cleanup
    if claude_client:
        await claude_client.cleanup()
    if local_client:
        await local_client.cleanup()
    print("👋 Shutdown complete!")


# FastAPI app
app = FastAPI(
    title="MCP Client API",
    description="API wrapper for MCP Client",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MCP Client API is running!",
        "status": "Server is up",
        "initialization": initialization_status,
        "endpoints": {
            "POST /query": "Process a query using MCP client",
            "GET /health": "Check API health",
            "GET /status": "Check initialization status",
            "GET /docs": "API documentation"
        }
    }


@app.get("/status")
async def get_status():
    """Get initialization status"""
    return {
        "initialization_status": initialization_status,
        "claude_ready": claude_client is not None and claude_client.is_connected,
        "local_ready": local_client is not None and local_client.is_connected
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server_running": True,
        "clients": {
            "claude_client_connected": claude_client is not None and claude_client.is_connected,
            "local_client_connected": local_client is not None and local_client.is_connected,
        }
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using the MCP client"""
    global claude_client, local_client

    try:
        print(f"🔍 Processing query: {request.query[:50]}...")

        # Check if clients are ready
        client = local_client if request.use_local_model else claude_client

        if not client or not client.is_connected:
            # Return status instead of error
            return QueryResponse(
                response="",
                error=f"MCP client not ready yet. Status: {initialization_status}"
            )

        print("🚀 Calling client.process_query...")
        # Process the query
        response = await client.process_query(request.query, request.local_model)
        print(f"✅ Got response of length: {len(response)} characters")
        print(f"📄 Response preview: {response[:100]}...")

        return QueryResponse(response=response)

    except Exception as e:
        print(f"❌ Error in process_query endpoint: {str(e)}")
        return QueryResponse(response="", error=str(e))


if __name__ == "__main__":
    print("Starting MCP Client API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)