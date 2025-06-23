"""
External MCP Manager

This module handles the connection to and management of external MCP servers.
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger('alita.external_mcp_manager')

class ExternalMCPManager:
    """
    Manages connections to external MCP servers and provides their tools.
    """
    def __init__(self, config_file: str = "mcp_servers.json"):
        self.config_file = config_file
        self.servers: List[Dict[str, Any]] = []
        self.tools: List[BaseTool] = []
        self.client: Optional[MultiServerMCPClient] = None
        self._load_config()

    def _load_config(self):
        """Loads server configurations from the JSON file."""
        if not os.path.exists(self.config_file):
            logger.warning(f"MCP server config file not found: {self.config_file}. Creating example file.")
            # Create an empty file with the new example structure
            example_config = {
              "mcpServers": {
                "mcp-github-trending": {
                  "transport": "stdio",
                  "command": "uvx",
                  "args": [
                      "mcp-github-trending"
                  ]
                }
              }
            }
            with open(self.config_file, 'w') as f:
                json.dump(example_config, f, indent=2)

        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            mcp_servers_config = config_data.get("mcpServers", {})
            self.servers = []
            for name, server_config in mcp_servers_config.items():
                full_server_config = {"name": name, **server_config}
                self.servers.append(full_server_config)

            logger.info(f"Loaded {len(self.servers)} MCP servers from {self.config_file}.")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load MCP server config: {e}")

    async def initialize_client(self):
        """Initializes the MultiServerMCPClient and fetches tools."""
        if not self.servers:
            logger.info("No enabled external MCP servers to initialize.")
            return

        try:
            # MultiServerMCPClient expects a dict of server configs, not a list
            # Build a dict keyed by server name
            servers_dict = {s["name"]: s for s in self.servers}
            self.client = MultiServerMCPClient(servers_dict)
            
            # Get tools directly without context manager
            tools = await self.client.get_tools()
            if not tools or not isinstance(tools, list):
                logger.warning("No tools returned or unexpected type from MultiServerMCPClient.get_tools().")
                self.tools = []
            else:
                self.tools = tools
            
            for tool in self.tools:
                # Add server name to tool description for clarity
                server_name = self._get_server_name_for_tool(tool)
                tool.description = f"[{server_name}] {tool.description}"

            logger.info(f"Successfully fetched {len(self.tools)} tools from {len(self.servers)} servers.")
        except Exception as e:
            logger.error(f"Failed to initialize MultiServerMCPClient or fetch tools: {e}")
            self.tools = []
            self.client = None
    
    def get_external_tools(self) -> List[BaseTool]:
        """Returns the list of fetched external tools."""
        return self.tools

    def _get_server_name_for_tool(self, tool: BaseTool) -> str:
        """Finds the server name associated with a given tool."""
        # This logic intrudes on the private interface of langchain_mcp_adapters,
        # but it's a reliable way to get the server definition for a tool.
        if not hasattr(tool, 'func') or not hasattr(tool.func, '__self__'):
            return "Unknown Server"
        
        # tool.func.__self__ should be an instance of langchain_mcp_adapters.client._Tool
        tool_adapter = getattr(tool.func, '__self__', None)
        if not hasattr(tool_adapter, 'client'):
            return "Unknown Server"
            
        # tool_adapter.client is an MCPClient instance
        mcp_client = getattr(tool_adapter, 'client', None)
        if not hasattr(mcp_client, 'server_definition'):
            return "Unknown Server"
        
        # The server_definition is the dictionary we passed into MultiServerMCPClient
        server_def = getattr(mcp_client, 'server_definition', {})
        return server_def.get("name", "Unknown Server")

    async def shutdown(self):
        """Shuts down the client connection."""
        if self.client:
            # For the new API, we don't need to explicitly shut down
            # The client will handle cleanup automatically
            logger.info("External MCP client shut down.")
            self.client = None

async def main():
    """Example usage of the ExternalMCPManager."""
    logging.basicConfig(level=logging.INFO)
    manager = ExternalMCPManager()
    await manager.initialize_client()
    
    tools = manager.get_external_tools()
    if tools:
        print("Available External Tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
    else:
        print("No external tools found.")

    await manager.shutdown()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 