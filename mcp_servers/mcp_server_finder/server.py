"""
MCP Server: mcp_server_finder
Tool to search and validate active MCP (Minecraft Protocol) servers that support Discord integration
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        TextContent,
        Tool,
    )
    MCP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MCP package not available: {e}. Using fallback implementation.")
    MCP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mcp_server_finder_mcp_server')

# Import the user's function
# MCP Name: mcp_server_finder
# Description: Tool to search and validate active MCP (Minecraft Protocol) servers that support Discord integration
# Arguments: query_params (dict): Dictionary containing search parameters (optional)
# Returns: List of dictionaries containing server information (ip, port, version, discord_support)
# Requires: mcstatus, requests, socket

import socket
import json
from typing import List, Dict, Optional
from mcstatus import JavaServer
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

def mcp_server_finder(query_params: Optional[Dict] = None) -> List[Dict]:
    """
    Searches for and validates Minecraft servers with Discord integration support.
    
    Args:
        query_params: Optional dictionary containing search parameters
                     (version, region, max_results, etc.)
    
    Returns:
        List of dictionaries containing validated server information
    """
    try:
        # Default parameters if none provided
        default_params = {
            "max_results": 10,
            "timeout": 5,
            "min_players": 0,
            "verify_discord": True
        }
        params = {**default_params, **(query_params or {})}
        
        # Initialize results list
        validated_servers = []
        
        # Sample server discovery (in production, would connect to server listing API)
        potential_servers = _discover_servers(params)
        
        # Validate servers in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_server = {
                executor.submit(_validate_server, server, params["timeout"]): server 
                for server in potential_servers
            }
            
            for future in future_to_server:
                server = future_to_server[future]
                try:
                    result = future.result()
                    if result:
                        validated_servers.append(result)
                        if len(validated_servers) >= params["max_results"]:
                            break
                except Exception:
                    continue
        
        return validated_servers

    except Exception as e:
        return [{"error": f"Server finder error: {str(e)}"}]

def _discover_servers(params: Dict) -> List[Dict]:
    """
    Discovers potential Minecraft servers (placeholder implementation).
    In production, would connect to server listing APIs.
    """
    # Placeholder server list - in production would fetch from API
    return [
        {"ip": "mc.example.com", "port": 25565},
        {"ip": "play.example.net", "port": 25565},
    ]

def _validate_server(server: Dict, timeout: int) -> Optional[Dict]:
    """
    Validates a single server's availability and Discord integration.
    """
    try:
        # Create JavaServer instance
        mc_server = JavaServer(server["ip"], server["port"])
        
        # Query server status
        status = mc_server.status()
        
        # Check for Discord integration via server MOTD or plugins
        has_discord = _check_discord_integration(status)
        
        return {
            "ip": server["ip"],
            "port": server["port"],
            "version": status.version.name,
            "players_online": status.players.online,
            "max_players": status.players.max,
            "latency": status.latency,
            "discord_support": has_discord,
            "last_checked": datetime.utcnow().isoformat()
        }
    except Exception:
        return None

def _check_discord_integration(status) -> bool:
    """
    Checks if server has Discord integration by analyzing MOTD and plugins.
    """
    # Check MOTD for Discord mentions
    motd_lower = status.description.lower() if hasattr(status, 'description') else ""
    if "discord" in motd_lower:
        return True
    
    # Check for common Discord integration plugins
    # This would be more comprehensive in production
    plugin_indicators = ["discordsrv", "discord_integration"]
    
    # In production, would check server's plugin list if available
    return any(plugin in motd_lower for plugin in plugin_indicators)

if MCP_AVAILABLE:
    # Create MCP server using the official package
    server = Server("mcp_server_finder")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools"""
        return [
            Tool(
                name="mcp_server_finder",
                description="Tool to search and validate active MCP (Minecraft Protocol) servers that support Discord integration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Input query for the tool"
                        }
                    },
                    "required": ["query"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Call a tool by name"""
        try:
            if name == "mcp_server_finder":
                query = arguments.get("query", "")
                logger.info(f"Calling {name} with query: {query}")
                
                # Call the user's function
                result = mcp_server_finder(query)
                
                return [
                    TextContent(
                        type="text",
                        text=str(result)
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )
                ]
        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}")
            return [
                TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )
            ]

    # Create initialization options and run server
    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)

else:
    # Fallback implementation using simple stdio
    class SimpleMCPServer:
        """Simple MCP server implementation"""
        
        def __init__(self, name: str):
            self.name = name
            self.tools = {
                "mcp_server_finder": {
                    "name": "mcp_server_finder",
                    "description": "Tool to search and validate active MCP (Minecraft Protocol) servers that support Discord integration",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Input query for the tool"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        
        async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
            """Handle MCP requests"""
            method = request.get("method")
            
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": self.name,
                            "version": "1.0.0"
                        }
                    }
                }
            
            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "tools": list(self.tools.values())
                    }
                }
            
            elif method == "tools/call":
                params = request.get("params", {})
                name = params.get("name")
                arguments = params.get("arguments", {})
                
                if name == "mcp_server_finder":
                    try:
                        query = arguments.get("query", "")
                        logger.info(f"Calling {name} with query: {query}")
                        
                        # Call the user's function
                        result = mcp_server_finder(query)
                        
                        return {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": str(result)
                                    }
                                ]
                            }
                        }
                    except Exception as e:
                        logger.error(f"Error calling tool {name}: {e}")
                        return {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Error: {str(e)}"
                                    }
                                ]
                            }
                        }
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Unknown tool: {name}"
                                }
                            ]
                        }
                    }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }

    async def main():
        """Main entry point using fallback implementation"""
        server = SimpleMCPServer("mcp_server_finder")
        
        # Simple stdio-based server
        import sys
        
        async def handle_stdio():
            while True:
                try:
                    line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                    if not line:
                        break
                    
                    request = json.loads(line.strip())
                    response = await server.handle_request(request)
                    
                    await asyncio.get_event_loop().run_in_executor(None, lambda: sys.stdout.write(json.dumps(response) + "\n"))
                    await asyncio.get_event_loop().run_in_executor(None, sys.stdout.flush)
                    
                except Exception as e:
                    logger.error(f"Error handling request: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id") if 'request' in locals() else None,
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        }
                    }
                    await asyncio.get_event_loop().run_in_executor(None, lambda: sys.stdout.write(json.dumps(error_response) + "\n"))
                    await asyncio.get_event_loop().run_in_executor(None, sys.stdout.flush)
        
        await handle_stdio()

    if __name__ == "__main__":
        asyncio.run(main())
