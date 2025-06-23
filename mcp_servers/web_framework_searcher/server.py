"""
MCP Server: web_framework_searcher
Searches and aggregates information about Python web frameworks from reliable sources
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
logger = logging.getLogger('web_framework_searcher_mcp_server')

# Import the user's function
# MCP Name: web_framework_searcher
# Description: Searches and aggregates information about Python web frameworks from reliable sources
# Arguments: query (str, optional) - specific framework name or search term
# Returns: dict containing framework information (name, description, github_url, stars, last_updated)
# Requires: requests, beautifulsoup4, PyGithub

import requests
from bs4 import BeautifulSoup
from github import Github
from datetime import datetime
import json
from typing import Dict, List, Optional
import os

def web_framework_searcher(query: Optional[str] = None) -> Dict:
    try:
        # Initialize results dictionary
        frameworks_data = {}
        
        # GitHub API token from environment variable
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            raise ValueError("GitHub token not found in environment variables")
        
        g = Github(github_token)
        
        # List of popular Python web frameworks to search
        frameworks = {
            "django": "django/django",
            "flask": "pallets/flask",
            "fastapi": "tiangolo/fastapi",
            "pyramid": "Pylons/pyramid",
            "tornado": "tornadoweb/tornado",
            "aiohttp": "aio-libs/aiohttp"
        }
        
        # Filter frameworks based on query if provided
        if query:
            frameworks = {k: v for k, v in frameworks.items() 
                        if query.lower() in k.lower()}
        
        # Fetch data for each framework
        for name, repo_path in frameworks.items():
            try:
                # Get GitHub repository information
                repo = g.get_repo(repo_path)
                
                # Get PyPI information
                pypi_url = f"https://pypi.org/pypi/{name}/json"
                pypi_response = requests.get(pypi_url)
                pypi_data = pypi_response.json() if pypi_response.status_code == 200 else {}
                
                frameworks_data[name] = {
                    "name": name,
                    "description": repo.description,
                    "github_url": repo.html_url,
                    "stars": repo.stargazers_count,
                    "last_updated": repo.updated_at.isoformat(),
                    "version": pypi_data.get("info", {}).get("version", "N/A"),
                    "python_version": pypi_data.get("info", {}).get("requires_python", "N/A"),
                    "license": repo.license.name if repo.license else "N/A",
                    "open_issues": repo.open_issues_count,
                    "homepage": repo.homepage or "N/A"
                }
            except Exception as e:
                frameworks_data[name] = {
                    "name": name,
                    "error": f"Failed to fetch data: {str(e)}"
                }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "frameworks": frameworks_data,
            "total_results": len(frameworks_data)
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "failed"
        }

if MCP_AVAILABLE:
    # Create MCP server using the official package
    server = Server("web_framework_searcher")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools"""
        return [
            Tool(
                name="web_framework_searcher",
                description="Searches and aggregates information about Python web frameworks from reliable sources",
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
            if name == "web_framework_searcher":
                query = arguments.get("query", "")
                logger.info(f"Calling {name} with query: {query}")
                
                # Call the user's function
                result = web_framework_searcher(query)
                
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
                "web_framework_searcher": {
                    "name": "web_framework_searcher",
                    "description": "Searches and aggregates information about Python web frameworks from reliable sources",
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
                
                if name == "web_framework_searcher":
                    try:
                        query = arguments.get("query", "")
                        logger.info(f"Calling {name} with query: {query}")
                        
                        # Call the user's function
                        result = web_framework_searcher(query)
                        
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
        server = SimpleMCPServer("web_framework_searcher")
        
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
