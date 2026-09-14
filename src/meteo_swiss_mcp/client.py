import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
import requests
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient, ChatResponse

from . import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP Client")
    parser.add_argument("--model", type=str, required=True,
                        help="LLM model name (e.g. 'qwen3:4b')")
    return parser.parse_args()

def _ensure_ollama() -> None:
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        # Ping Ollama API
        requests.get(f"{ollama_base_url}/api/tags", timeout=1)
        logger.info("Ollama running.")
    except Exception:
        logger.info("Starting Ollama server...")
        # Start Ollama, do not write to the console
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait until server is ready
        for _ in range(20):
            try:
                requests.get(f"{ollama_base_url}/api/tags", timeout=1)
                logger.info("Ollama started.")
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("Failed to start Ollama within timeout")

class MCPClient:
    def __init__(self, model: str):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.ollama_client = AsyncClient()
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None

    async def connect_to_server(self) -> None:
        # Launch the server module with the same interpreter running this client
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "meteo_swiss_mcp.server"],
        )

        # Connect to the server
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        # Initialize the connection
        await self.session.initialize()

        # List available tools
        tools_result = await self.session.list_tools()
        logger.info("Connected to server with tools:")
        for tool in tools_result.tools:
            logger.info(f"  - {tool.name}: {tool.description}")

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            }
            for tool in tools_result.tools
        ]
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        logger.info(f"Tool: {tool_name} called with arguments: {arguments}")
        try:
            response = await self.session.call_tool(tool_name, arguments)
            return response
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return f"Error calling tool {tool_name}: {str(e)}"

    def _format_assistant_message(self, response: ChatResponse) -> Dict[str, Any]:
        """
        Standardizes assistant messages into a clean dictionary format.
        Preserves tool_calls while removing internal metadata (images, thinking, etc).
        """
        message = response.message
        formatted_message = {
            "role": "assistant",
            "content": message.content or ""
        }

        # If the model wants to use tools, add them to the dictionary
        if message.tool_calls:
            formatted_message["tool_calls"] = [
                {
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                }
                for call in message.tool_calls
            ]

        return formatted_message

    async def process_query(self, query: str) -> str:
        self.messages.append({"role": "user", "content": query})
        tools = await self.get_mcp_tools()

        while True:
            response = await self.ollama_client.chat(model=self.model, messages=self.messages, tools=tools)
            assistant_message = self._format_assistant_message(response)
            self.messages.append(assistant_message)

            # If no tools were requested, we have the final answer
            if not response.message.tool_calls:
                return response.message.content

            # Process each tool call
            for tool_call in response.message.tool_calls:
                tool_call_response = await self.call_tool(tool_call.function.name, tool_call.function.arguments)
                self.messages.append({
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": tool_call_response.content[0].text,
                })

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    _ensure_ollama()
    args = _parse_args()
    client = MCPClient(args.model)
    await client.connect_to_server()
    try:
        logger.info(f"MCP Client started!")
        loop = asyncio.get_running_loop()
        while True:
            query = await loop.run_in_executor(None, input, "\nType your question (or '/bye' to quit): ")
            query = query.strip()
            if query.lower() == "/bye":
                logger.info(f"Goodbye!")
                break

            if not query:
                logger.info(f"Please enter a non-empty query.")
                continue

            logger.info(f"Query: {query}")
            response = await client.process_query(query)
            logger.info(f"Response: {response}")

    except KeyboardInterrupt:
        logger.info(f"Interrupted by user, shutting down...")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())