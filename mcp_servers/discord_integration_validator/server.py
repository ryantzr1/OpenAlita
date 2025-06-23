"""
MCP Server: discord_integration_validator
Validates Discord bot permissions and integration capabilities for found MCP servers
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
logger = logging.getLogger('discord_integration_validator_mcp_server')

# Import the user's function
# MCP Name: discord_integration_validator
# Description: Validates Discord bot permissions and integration capabilities for found MCP servers
# Arguments: server_config (dict): Configuration containing Discord integration details
# Returns: dict containing validation results with status, permissions check, and detailed messages
# Requires: discord.py, aiohttp, typing

import discord
from typing import Dict, Any, List
import asyncio
import aiohttp
from datetime import datetime

async def _check_bot_connection(token: str) -> tuple[bool, str]:
    """Test Discord bot token connectivity."""
    try:
        client = discord.Client(intents=discord.Intents.default())
        await client.login(token)
        await client.close()
        return True, "Bot token is valid and can connect"
    except discord.LoginFailure:
        return False, "Invalid bot token"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def _validate_permissions(permissions: int) -> tuple[bool, List[str]]:
    """Validate required bot permissions."""
    required_permissions = {
        'send_messages': 0x800,
        'read_messages': 0x400,
        'manage_webhooks': 0x20,
        'embed_links': 0x4000
    }
    
    missing_permissions = []
    for perm_name, perm_value in required_permissions.items():
        if not (permissions & perm_value):
            missing_permissions.append(perm_name)
    
    return len(missing_permissions) == 0, missing_permissions

async def discord_integration_validator(server_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates Discord bot permissions and integration capabilities for MCP servers.
    
    Args:
        server_config: Dictionary containing Discord integration configuration
            Required keys: 'bot_token', 'permissions', 'client_id'
    """
    try:
        # Validate configuration structure
        required_fields = ['bot_token', 'permissions', 'client_id']
        missing_fields = [field for field in required_fields if field not in server_config]
        
        if missing_fields:
            return {
                'status': 'error',
                'valid': False,
                'message': f"Missing required fields: {', '.join(missing_fields)}",
                'timestamp': datetime.utcnow().isoformat()
            }

        # Validate bot token and connection
        connection_valid, connection_message = await _check_bot_connection(server_config['bot_token'])
        
        # Validate permissions
        permissions_valid, missing_permissions = _validate_permissions(int(server_config['permissions']))

        # Validate client ID format
        client_id_valid = server_config['client_id'].isdigit() and len(server_config['client_id']) > 16

        # Compile validation results
        validation_result = {
            'status': 'success' if all([connection_valid, permissions_valid, client_id_valid]) else 'error',
            'valid': all([connection_valid, permissions_valid, client_id_valid]),
            'details': {
                'connection': {
                    'valid': connection_valid,
                    'message': connection_message
                },
                'permissions': {
                    'valid': permissions_valid,
                    'missing_permissions': missing_permissions if not permissions_valid else []
                },
                'client_id': {
                    'valid': client_id_valid,
                    'message': 'Valid client ID' if client_id_valid else 'Invalid client ID format'
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        }

        return validation_result

    except Exception as e:
        return {
            'status': 'error',
            'valid': False,
            'message': f"Validation error: {str(e)}",
            'timestamp': datetime.utcnow().isoformat()
        }

# Example usage:
# config = {
#     'bot_token': 'your-bot-token',
#     'permissions': 8072,
#     'client_id': '123456789012345678'
# }
# result = asyncio.run(discord_integration_validator(config))

if MCP_AVAILABLE:
    # Create MCP server using the official package
    server = Server("discord_integration_validator")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools"""
        return [
            Tool(
                name="discord_integration_validator",
                description="Validates Discord bot permissions and integration capabilities for found MCP servers",
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
            if name == "discord_integration_validator":
                query = arguments.get("query", "")
                logger.info(f"Calling {name} with query: {query}")
                
                # Call the user's function
                result = discord_integration_validator(query)
                
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
                "discord_integration_validator": {
                    "name": "discord_integration_validator",
                    "description": "Validates Discord bot permissions and integration capabilities for found MCP servers",
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
                
                if name == "discord_integration_validator":
                    try:
                        query = arguments.get("query", "")
                        logger.info(f"Calling {name} with query: {query}")
                        
                        # Call the user's function
                        result = discord_integration_validator(query)
                        
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
        server = SimpleMCPServer("discord_integration_validator")
        
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
