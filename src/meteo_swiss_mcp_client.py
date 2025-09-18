import argparse
import asyncio
import json
import logging
import subprocess
import time
import requests
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

# Configure logger
def _setup_logger() -> None:
    try:
        config_path = Path(__file__).parent.parent / "log_config.json"
        with open(config_path, "rt") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
        logging.getLogger("meteo_swiss_mcp_client").info("Logger successfully configured")
    except Exception as e:
        logging.basicConfig(level=logging.ERROR) # logging.basicConfig() attaches the root logger to STDERR by default, so no interference with stdio MCP protocol
        logging.getLogger("meteo_swiss_mcp_client").error(f"Failed to configure logger: {e}", exc_info=True)


_setup_logger()
logger = logging.getLogger("meteo_swiss_mcp_client")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP Client")
    parser.add_argument("--model", type=str, required=True,
                        help="LLM model name (e.g. 'qwen3:4b')")
    parser.add_argument("--server-script", type=str, required=True,
                        help="Path to server script (.py)")
    args = parser.parse_args()
    if not Path(args.server_script).exists():
        parser.error(f"Server script {args.server_script} does not exist")
    return args

def _ensure_ollama() -> None:
    try:
        # Ping Ollama API
        requests.get("http://localhost:11434/api/tags", timeout=1)
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
                requests.get("http://localhost:11434/api/tags", timeout=1)
                print("Ollama started.")
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
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None

    async def connect_to_server(self, server_script_path: str) -> None:
        # Server configuration
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
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

    async def process_query(self, query: str) -> str:
        tools = await self.get_mcp_tools()

        response = await self.ollama_client.chat(
            model=self.model,
            messages=[{"role": "user", "content": query}],
            tools=tools,
        )

        # Get assistant's response
        assistant_message = response.message

        # Initialize conversation with user query and assistant response
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant_message.content}
        ]

        # Handle tool calls if present
        if assistant_message.tool_calls:
            # Process each tool call
            for tool_call in assistant_message.tool_calls:
                # Execute tool call
                tool_call_response = await self.call_tool(tool_call.function.name, tool_call.function.arguments)

                # Add tool response to conversation
                messages.append({ "role": "tool", "content": tool_call_response.content[0].text})

            # Get final response using information from tools
            final_response = await self.ollama_client.chat(
                model=self.model,
                messages=messages,
                tools=tools
            )

            return final_response.message.content

        # No tool calls, just return the direct response
        return assistant_message.content

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    _ensure_ollama()
    args = _parse_args()
    client = MCPClient(args.model)
    await client.connect_to_server(args.server_script)
    try:
        logger.info(f"MCP Client started!")
        while True:
            query = input("\nType your question (or '/bye' to quit): ").strip()
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