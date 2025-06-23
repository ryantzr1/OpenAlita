"""
MCP Server: weather_data_fetcher
Fetches weather data from reliable weather APIs
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
logger = logging.getLogger('weather_data_fetcher_mcp_server')

# Import the user's function
# MCP Name: weather_data_fetcher
# Description: Fetches weather data from reliable weather APIs
# Arguments: city: str - name of the city, api_key: str - OpenWeatherMap API key
# Returns: Dictionary containing current weather data including temperature, humidity, and conditions
# Requires: requests, python-dotenv

import os
import requests
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

def weather_data_fetcher(city: str, api_key: str = None) -> Dict[str, Any]:
    """
    Fetches current weather data for a specified city using OpenWeatherMap API.
    
    Args:
        city: Name of the city to fetch weather data for
        api_key: OpenWeatherMap API key (optional if set in environment variables)
    
    Returns:
        Dictionary containing weather data or error message
    """
    try:
        # Load environment variables if API key not provided
        if not api_key:
            load_dotenv()
            api_key = os.getenv('OPENWEATHER_API_KEY')
            if not api_key:
                raise ValueError("API key not provided and not found in environment variables")

        # API endpoint configuration
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric"  # Use metric units for temperature
        }

        # Make API request
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract relevant weather information
        weather_data = {
            "timestamp": datetime.now().isoformat(),
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": {
                "current": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "min": data["main"]["temp_min"],
                "max": data["main"]["temp_max"]
            },
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "weather_condition": {
                "main": data["weather"][0]["main"],
                "description": data["weather"][0]["description"]
            },
            "wind": {
                "speed": data["wind"]["speed"],
                "direction": data["wind"].get("deg", None)
            },
            "clouds": data["clouds"]["all"]
        }

        return weather_data

    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except KeyError as e:
        return {"error": f"Data parsing error: {str(e)}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

if MCP_AVAILABLE:
    # Create MCP server using the official package
    server = Server("weather_data_fetcher")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools"""
        return [
            Tool(
                name="weather_data_fetcher",
                description="Fetches weather data from reliable weather APIs",
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
            if name == "weather_data_fetcher":
                query = arguments.get("query", "")
                logger.info(f"Calling {name} with query: {query}")
                
                # Call the user's function
                result = weather_data_fetcher(query)
                
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
                "weather_data_fetcher": {
                    "name": "weather_data_fetcher",
                    "description": "Fetches weather data from reliable weather APIs",
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
                
                if name == "weather_data_fetcher":
                    try:
                        query = arguments.get("query", "")
                        logger.info(f"Calling {name} with query: {query}")
                        
                        # Call the user's function
                        result = weather_data_fetcher(query)
                        
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
        server = SimpleMCPServer("weather_data_fetcher")
        
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
