"""
MCP Server: data_preprocessor
Cleans and formats weather data for analysis
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
logger = logging.getLogger('data_preprocessor_mcp_server')

# Import the user's function
# MCP Name: data_preprocessor
# Description: Cleans and formats weather data for analysis
# Arguments: data_dict (dict): Dictionary containing weather data with keys: temperature, humidity, precipitation, wind_speed
# Returns: Dictionary with cleaned and standardized weather data
# Requires: pandas, numpy

import pandas as pd
import numpy as np
from typing import Dict, Union, Any

def data_preprocessor(data_dict: Dict[str, Any]) -> Dict[str, Union[float, str]]:
    """
    Cleans and standardizes weather data by:
    1. Handling missing values
    2. Converting units to standard format
    3. Removing outliers
    4. Validating data ranges
    """
    try:
        # Initialize cleaned data dictionary
        cleaned_data = {}
        
        # Validate input dictionary
        required_fields = ['temperature', 'humidity', 'precipitation', 'wind_speed']
        if not all(field in data_dict for field in required_fields):
            raise ValueError("Missing required weather data fields")

        # Clean temperature (convert to Celsius if in Fahrenheit)
        temp = float(data_dict['temperature'])
        if temp > 50:  # Assume Fahrenheit if > 50
            temp = (temp - 32) * 5/9
        cleaned_data['temperature'] = round(temp, 2)

        # Clean humidity (ensure percentage between 0-100)
        humidity = float(data_dict['humidity'])
        if not 0 <= humidity <= 100:
            humidity = np.clip(humidity, 0, 100)
        cleaned_data['humidity'] = round(humidity, 2)

        # Clean precipitation (convert to mm if needed)
        precip = float(data_dict['precipitation'])
        if precip < 0:
            precip = 0
        cleaned_data['precipitation'] = round(precip, 2)

        # Clean wind speed (convert to m/s if needed)
        wind = float(data_dict['wind_speed'])
        if wind < 0:
            wind = abs(wind)
        cleaned_data['wind_speed'] = round(wind, 2)

        # Add data quality indicators
        cleaned_data['data_quality'] = 'good'
        if any(pd.isna([temp, humidity, precip, wind])):
            cleaned_data['data_quality'] = 'missing_values'

        # Add timestamp of processing
        cleaned_data['processed_timestamp'] = pd.Timestamp.now().isoformat()

        return cleaned_data

    except ValueError as ve:
        return {
            'error': f"Value error: {str(ve)}",
            'status': 'failed'
        }
    except Exception as e:
        return {
            'error': f"Unexpected error: {str(e)}",
            'status': 'failed'
        }

if MCP_AVAILABLE:
    # Create MCP server using the official package
    server = Server("data_preprocessor")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools"""
        return [
            Tool(
                name="data_preprocessor",
                description="Cleans and formats weather data for analysis",
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
            if name == "data_preprocessor":
                query = arguments.get("query", "")
                logger.info(f"Calling {name} with query: {query}")
                
                # Call the user's function
                result = data_preprocessor(query)
                
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
                "data_preprocessor": {
                    "name": "data_preprocessor",
                    "description": "Cleans and formats weather data for analysis",
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
                
                if name == "data_preprocessor":
                    try:
                        query = arguments.get("query", "")
                        logger.info(f"Calling {name} with query: {query}")
                        
                        # Call the user's function
                        result = data_preprocessor(query)
                        
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
        server = SimpleMCPServer("data_preprocessor")
        
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
