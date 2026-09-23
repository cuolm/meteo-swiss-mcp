import argparse
import asyncio
import json
import logging
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError as error:
    raise SystemExit(
        "The MCP client needs the OpenAI SDK, which ships in the optional 'client' extra.\n"
        "Install it with one of:\n"
        "    uv tool install 'swiss-weather-mcp[client]'   # installed as a tool\n"
        "    uv sync --extra client                        # from a source checkout\n"
        "    pip install 'swiss-weather-mcp[client]'"
    ) from error

from . import LOG_LEVELS, setup_logging

logger = logging.getLogger(__name__)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP Client")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name as the server reports it")
    parser.add_argument("--base-url", type=str, default="http://localhost:8080/v1",
                        help="OpenAI compatible endpoint of an already running server (default: %(default)s)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging level for the client (default: INFO)",
    )
    return parser.parse_args()

class MCPClient:
    def __init__(self, model: str, base_url: str):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm_client = AsyncOpenAI(base_url=base_url, api_key="not-needed")  # local servers ignore the key
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        self.read_stream: Optional[Any] = None
        self.write_stream: Optional[Any] = None

    async def connect_to_server(self) -> None:
        # Launch the server module with the same interpreter running this client
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "swiss_weather_mcp.server"],
        )

        # Connect to the server
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.read_stream, self.write_stream = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read_stream, self.write_stream)
        )

        # Initialize the connection
        await self.session.initialize()

        # List available tools, names only, the descriptions are multi line docstrings
        tools_result = await self.session.list_tools()
        tool_names = ", ".join(tool.name for tool in tools_result.tools)
        logger.info(f"Connected to server with tools: {tool_names}")

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            }
            for tool in tools_result.tools
        ]
    
    async def call_tool(self, tool_name: str, arguments_json: str) -> str:
        logger.info(f"Tool: {tool_name} called with arguments: {arguments_json}")
        try:
            arguments = json.loads(arguments_json or "{}")  # the model writes these as JSON text
            response = await self.session.call_tool(tool_name, arguments)
            return response.content[0].text
        except Exception as error:
            logger.error(f"Tool {tool_name} failed: {error}")
            return f"Error calling tool {tool_name}: {error}"

    def _format_assistant_message(self, message: Any) -> Dict[str, Any]:
        """
        Standardizes assistant messages into a clean dictionary format.
        Preserves tool_calls while removing internal metadata (images, thinking, etc).
        """
        formatted_message = {
            "role": "assistant",
            "content": message.content or ""
        }

        # If the model wants to use tools, add them to the dictionary
        if message.tool_calls:
            formatted_message["tool_calls"] = [
                {
                    "id": tool_call.id,  # each result has to reference the call it answers
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                }
                for tool_call in message.tool_calls
            ]

        return formatted_message

    async def process_query(self, query: str) -> str:
        self.messages.append({"role": "user", "content": query})
        tools = await self.get_mcp_tools()

        while True:
            response = await self.llm_client.chat.completions.create(
                model=self.model, messages=self.messages, tools=tools
            )
            message = response.choices[0].message
            self.messages.append(self._format_assistant_message(message))

            # If no tools were requested, we have the final answer
            if not message.tool_calls:
                return message.content

            # Process each tool call
            for tool_call in message.tool_calls:
                tool_result = await self.call_tool(tool_call.function.name, tool_call.function.arguments)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

    async def cleanup(self):
        await self.exit_stack.aclose()


async def _run():
    args = _parse_args()
    setup_logging(args.log_level)
    client = MCPClient(args.model, args.base_url)
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

def main():
    asyncio.run(_run())

if __name__ == "__main__":
    main()