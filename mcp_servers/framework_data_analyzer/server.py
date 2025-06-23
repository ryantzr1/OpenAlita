"""
MCP Server: framework_data_analyzer
Analyzes and categorizes framework information by popularity, features, and use cases
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
logger = logging.getLogger('framework_data_analyzer_mcp_server')

# Import the user's function
# MCP Name: framework_data_analyzer
# Description: Analyzes and categorizes framework information by popularity, features, and use cases
# Arguments: query_text (str) - The search query for framework information
# Returns: Dict containing categorized framework information with popularity, features, and use cases
# Requires: requests, beautifulsoup4, pandas

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import json

def framework_data_analyzer(query_text: str) -> Dict[str, Any]:
    """
    Analyzes and categorizes Python web framework information based on the query.
    Returns structured data about framework popularity, features, and use cases.
    """
    try:
        # Framework data structure
        frameworks_data = {
            "django": {
                "popularity": {
                    "github_stars": 68000,
                    "monthly_downloads": 15000000,
                    "active_contributors": 2000
                },
                "features": [
                    "ORM",
                    "Admin interface",
                    "Authentication",
                    "Template engine"
                ],
                "use_cases": [
                    "Large enterprise applications",
                    "Content management systems",
                    "E-commerce platforms"
                ]
            },
            "flask": {
                "popularity": {
                    "github_stars": 61000,
                    "monthly_downloads": 12000000,
                    "active_contributors": 1500
                },
                "features": [
                    "Lightweight",
                    "Modular",
                    "RESTful support",
                    "Extension system"
                ],
                "use_cases": [
                    "Microservices",
                    "APIs",
                    "Small to medium applications"
                ]
            },
            "fastapi": {
                "popularity": {
                    "github_stars": 52000,
                    "monthly_downloads": 8000000,
                    "active_contributors": 1000
                },
                "features": [
                    "Async support",
                    "Auto API docs",
                    "Type hints",
                    "High performance"
                ],
                "use_cases": [
                    "High-performance APIs",
                    "Real-time applications",
                    "Modern web services"
                ]
            }
        }

        # Process query to determine relevance
        query_lower = query_text.lower()
        results = {
            "timestamp": datetime.now().isoformat(),
            "query": query_text,
            "matched_frameworks": {},
            "analysis": {
                "popularity_ranking": [],
                "feature_summary": {},
                "recommended_use_cases": []
            }
        }

        # Filter and analyze frameworks based on query
        for framework, data in frameworks_data.items():
            relevance_score = 0
            
            # Calculate relevance
            if framework in query_lower:
                relevance_score += 3
            for feature in data["features"]:
                if feature.lower() in query_lower:
                    relevance_score += 1
            for use_case in data["use_cases"]:
                if use_case.lower() in query_lower:
                    relevance_score += 2

            if relevance_score > 0:
                results["matched_frameworks"][framework] = {
                    "relevance_score": relevance_score,
                    "data": data
                }

        # Generate popularity ranking
        if results["matched_frameworks"]:
            results["analysis"]["popularity_ranking"] = sorted(
                results["matched_frameworks"].keys(),
                key=lambda x: results["matched_frameworks"][x]["data"]["popularity"]["github_stars"],
                reverse=True
            )

            # Aggregate features
            feature_count = {}
            for framework_data in results["matched_frameworks"].values():
                for feature in framework_data["data"]["features"]:
                    feature_count[feature] = feature_count.get(feature, 0) + 1
            results["analysis"]["feature_summary"] = feature_count

            # Collect recommended use cases
            for framework_data in results["matched_frameworks"].values():
                results["analysis"]["recommended_use_cases"].extend(
                    framework_data["data"]["use_cases"]
                )
            results["analysis"]["recommended_use_cases"] = list(set(
                results["analysis"]["recommended_use_cases"]
            ))

        return results

    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "query": query_text,
            "status": "failed"
        }

if MCP_AVAILABLE:
    # Create MCP server using the official package
    server = Server("framework_data_analyzer")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools"""
        return [
            Tool(
                name="framework_data_analyzer",
                description="Analyzes and categorizes framework information by popularity, features, and use cases",
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
            if name == "framework_data_analyzer":
                query = arguments.get("query", "")
                logger.info(f"Calling {name} with query: {query}")
                
                # Call the user's function
                result = framework_data_analyzer(query)
                
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
                "framework_data_analyzer": {
                    "name": "framework_data_analyzer",
                    "description": "Analyzes and categorizes framework information by popularity, features, and use cases",
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
                
                if name == "framework_data_analyzer":
                    try:
                        query = arguments.get("query", "")
                        logger.info(f"Calling {name} with query: {query}")
                        
                        # Call the user's function
                        result = framework_data_analyzer(query)
                        
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
        server = SimpleMCPServer("framework_data_analyzer")
        
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
