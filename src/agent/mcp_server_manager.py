"""
MCP Server Manager

Handles creation, startup, and management of local MCP servers.
"""

import json
import logging
import os
import subprocess
import asyncio
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re

logger = logging.getLogger('alita.mcp_server_manager')

class MCPServerManager:
    """Manages local MCP server creation and execution"""
    
    def __init__(self, config_file: str = "mcp_servers.json", servers_dir: str = "mcp_servers"):
        self.config_file = config_file
        self.servers_dir = Path(servers_dir)
        self.servers_dir.mkdir(exist_ok=True)
        self.running_servers: Dict[str, subprocess.Popen] = {}
        self.server_ports: Dict[str, int] = {}
        self.next_port = 8001
        
    def _load_config(self) -> Dict[str, Any]:
        """Load current MCP server configuration"""
        if not os.path.exists(self.config_file):
            return {"mcpServers": {}}
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load MCP config: {e}")
            return {"mcpServers": {}}
    
    def _save_config(self, config: Dict[str, Any]):
        """Save MCP server configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Updated MCP server configuration: {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save MCP config: {e}")
    
    def _get_available_port(self) -> int:
        """Get next available port starting from 8001"""
        port = self.next_port
        while self._is_port_in_use(port):
            port += 1
        self.next_port = port + 1
        return port
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if port is in use"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return False
        except OSError:
            return True
    
    def create_mcp_server(self, server_name: str, script_content: str, 
                         metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """Create a new MCP server with the given script content."""
        try:
            logger.info(f"Creating MCP server: {server_name}")
            
            # Clean script content
            cleaned_script = self._clean_script_content(script_content)
            if not cleaned_script:
                logger.error(f"Failed to clean script content for {server_name}")
                return False, "Failed to clean script content"
            
            # Create server directory
            server_dir = self.servers_dir / server_name
            server_dir.mkdir(parents=True, exist_ok=True)
            
            # Create server.py file
            server_file = server_dir / "server.py"
            
            # Generate MCP server code
            server_code = self._generate_mcp_server_code(server_name, cleaned_script, metadata)
            
            with open(server_file, 'w') as f:
                f.write(server_code)
            
            # Create requirements.txt if needed
            requires = metadata.get('requires', '')
            if requires:
                requirements_file = server_dir / "requirements.txt"
                with open(requirements_file, 'w') as f:
                    f.write(requires)
            
            logger.info(f"Successfully created MCP server: {server_name}")
            return True, str(server_file)
            
        except Exception as e:
            logger.error(f"Failed to create MCP server {server_name}: {e}")
            return False, str(e)
    
    def _clean_script_content(self, script_content: str) -> str:
        """Clean LLM-generated script content to extract only valid Python code."""
        import re
        
        # Remove markdown code fences
        script = re.sub(r'```(?:python)?\s*', '', script_content)
        script = re.sub(r'```\s*', '', script)
        
        lines = script.split('\n')
        cleaned_lines = []
        in_python_code = False
        in_function = False
        function_depth = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Start of Python code section
            if (stripped.startswith('# MCP Name:') or 
                stripped.startswith('import ') or 
                stripped.startswith('from ') or
                stripped.startswith('def ') or
                stripped.startswith('class ') or
                stripped.startswith('@') or
                in_python_code):
                
                in_python_code = True
                
                # Track function definition and indentation
                if stripped.startswith('def '):
                    in_function = True
                    function_depth = 0
                
                # Track indentation level within function
                if in_function:
                    if stripped.endswith(':'):
                        function_depth += 1
                    elif stripped.startswith('return ') or stripped.startswith('if ') or stripped.startswith('elif ') or stripped.startswith('else:') or stripped.startswith('try:') or stripped.startswith('except ') or stripped.startswith('finally:') or stripped.startswith('for ') or stripped.startswith('while ') or stripped.startswith('with ') or stripped.startswith('async ') or stripped.startswith('await '):
                        # These are valid Python statements, continue
                        pass
                    elif stripped == '':
                        # Empty line is fine
                        pass
                    elif stripped.startswith('    ') or stripped.startswith('\t'):
                        # Indented code is fine
                        pass
                    elif stripped.startswith('#'):
                        # Comments are fine
                        pass
                    elif stripped.startswith('"""') or stripped.startswith("'''"):
                        # Docstrings are fine
                        pass
                    elif stripped.endswith('"""') or stripped.endswith("'''"):
                        # End of docstrings are fine
                        pass
                    elif stripped.startswith('_original_') or stripped.startswith('# Original function') or stripped.startswith('# Debug wrapper'):
                        # Stop at wrapper code
                        break
                    elif not any(stripped.startswith(valid) for valid in [
                        'import ', 'from ', 'def ', 'class ', '@', 'return ', 'if ', 'elif ', 'else:', 
                        'try:', 'except ', 'finally:', 'for ', 'while ', 'with ', 'async ', 'await ',
                        '#', '"""', "'''", '    ', '\t', ''
                    ]):
                        # If we hit explanatory text, stop
                        break
                
                cleaned_lines.append(line)
            elif stripped == '' and in_python_code:
                # Keep empty lines within Python code
                cleaned_lines.append(line)
        
        cleaned_script = '\n'.join(cleaned_lines).strip()
        
        # Validate that we have a function definition
        if 'def ' not in cleaned_script:
            logger.warning("No function definition found in cleaned script")
            return ""
        
        # Remove any trailing explanatory text that might have slipped through
        lines = cleaned_script.split('\n')
        final_lines = []
        for line in lines:
            stripped = line.strip()
            # Stop if we hit explanatory text
            if (stripped.startswith('This implementation') or 
                stripped.startswith('The function') or
                stripped.startswith('To use this') or
                stripped.startswith('The tool') or
                stripped.startswith('Example usage') or
                stripped.startswith('The implementation')):
                break
            final_lines.append(line)
        
        cleaned_script = '\n'.join(final_lines).strip()
        
        return cleaned_script
    
    def _generate_mcp_server_code(self, server_name: str, script_content: str, 
                                 metadata: Dict[str, Any]) -> str:
        """Generate MCP server code from script content"""
        
        # Extract function name from metadata or use server_name
        function_name = metadata.get('name', server_name)
        description = metadata.get('description', f'MCP server for {server_name}')
        
        # Create the MCP server code with proper MCP package imports
        server_code = f'''"""
MCP Server: {server_name}
{description}
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
    logging.warning(f"MCP package not available: {{e}}. Using fallback implementation.")
    MCP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('{server_name}_mcp_server')

# Import the user's function
{script_content}

if MCP_AVAILABLE:
    # Create MCP server using the official package
    server = Server("{server_name}")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools"""
        return [
            Tool(
                name="{function_name}",
                description="{description}",
                inputSchema={{
                    "type": "object",
                    "properties": {{
                        "query": {{
                            "type": "string",
                            "description": "Input query for the tool"
                        }}
                    }},
                    "required": ["query"]
                }}
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Call a tool by name"""
        try:
            if name == "{function_name}":
                query = arguments.get("query", "")
                logger.info(f"Calling {{name}} with query: {{query}}")
                
                # Call the user's function
                result = {function_name}(query)
                
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
                        text=f"Unknown tool: {{name}}"
                    )
                ]
        except Exception as e:
            logger.error(f"Error calling tool {{name}}: {{e}}")
            return [
                TextContent(
                    type="text",
                    text=f"Error: {{str(e)}}"
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
            self.tools = {{
                "{function_name}": {{
                    "name": "{function_name}",
                    "description": "{description}",
                    "inputSchema": {{
                        "type": "object",
                        "properties": {{
                            "query": {{
                                "type": "string",
                                "description": "Input query for the tool"
                            }}
                        }},
                        "required": ["query"]
                    }}
                }}
            }}
        
        async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
            """Handle MCP requests"""
            method = request.get("method")
            
            if method == "initialize":
                return {{
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {{
                        "protocolVersion": "2024-11-05",
                        "capabilities": {{
                            "tools": {{}}
                        }},
                        "serverInfo": {{
                            "name": self.name,
                            "version": "1.0.0"
                        }}
                    }}
                }}
            
            elif method == "tools/list":
                return {{
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {{
                        "tools": list(self.tools.values())
                    }}
                }}
            
            elif method == "tools/call":
                params = request.get("params", {{}})
                name = params.get("name")
                arguments = params.get("arguments", {{}})
                
                if name == "{function_name}":
                    try:
                        query = arguments.get("query", "")
                        logger.info(f"Calling {{name}} with query: {{query}}")
                        
                        # Call the user's function
                        result = {function_name}(query)
                        
                        return {{
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {{
                                "content": [
                                    {{
                                        "type": "text",
                                        "text": str(result)
                                    }}
                                ]
                            }}
                        }}
                    except Exception as e:
                        logger.error(f"Error calling tool {{name}}: {{e}}")
                        return {{
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {{
                                "content": [
                                    {{
                                        "type": "text",
                                        "text": f"Error: {{str(e)}}"
                                    }}
                                ]
                            }}
                        }}
                else:
                    return {{
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {{
                            "content": [
                                {{
                                    "type": "text",
                                    "text": f"Unknown tool: {{name}}"
                                }}
                            ]
                        }}
                    }}
            
            else:
                return {{
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {{
                        "code": -32601,
                        "message": f"Method not found: {{method}}"
                    }}
                }}

    async def main():
        """Main entry point using fallback implementation"""
        server = SimpleMCPServer("{server_name}")
        
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
                    
                    await asyncio.get_event_loop().run_in_executor(None, lambda: sys.stdout.write(json.dumps(response) + "\\n"))
                    await asyncio.get_event_loop().run_in_executor(None, sys.stdout.flush)
                    
                except Exception as e:
                    logger.error(f"Error handling request: {{e}}")
                    error_response = {{
                        "jsonrpc": "2.0",
                        "id": request.get("id") if 'request' in locals() else None,
                        "error": {{
                            "code": -32603,
                            "message": f"Internal error: {{str(e)}}"
                        }}
                    }}
                    await asyncio.get_event_loop().run_in_executor(None, lambda: sys.stdout.write(json.dumps(error_response) + "\\n"))
                    await asyncio.get_event_loop().run_in_executor(None, sys.stdout.flush)
        
        await handle_stdio()

    if __name__ == "__main__":
        asyncio.run(main())
'''
        
        return server_code
    
    def start_mcp_server(self, server_name: str, server_path: str) -> Tuple[bool, str]:
        """
        Start a local MCP server
        
        Returns:
            (success, port_or_error_message)
        """
        try:
            # Check if server is already running
            if server_name in self.running_servers:
                return True, f"Server {server_name} is already running"
            
            # Get available port
            port = self._get_available_port()
            self.server_ports[server_name] = port
            
            # Start server process
            server_file = Path(server_path) / "server.py"
            if not server_file.exists():
                return False, f"Server file not found: {server_file}"
            
            # Install dependencies if requirements.txt exists
            req_file = Path(server_path) / "requirements.txt"
            if req_file.exists():
                try:
                    subprocess.run([
                        "pip", "install", "-r", str(req_file)
                    ], check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install requirements for {server_name}: {e}")
            
            # Start the server process
            process = subprocess.Popen([
                "python", str(server_file)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.running_servers[server_name] = process
            
            # Wait a moment for server to start
            import time
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"Started MCP server {server_name} on port {port}")
                return True, str(port)
            else:
                # Process died, get error output
                stdout, stderr = process.communicate()
                error_msg = stderr.decode() if stderr else stdout.decode()
                return False, f"Server failed to start: {error_msg}"
                
        except Exception as e:
            logger.error(f"Failed to start MCP server {server_name}: {e}")
            return False, str(e)
    
    def add_server_to_config(self, server_name: str, server_path: str, port: int):
        """Add server to mcp_servers.json configuration"""
        config = self._load_config()
        
        # Add server configuration
        config["mcpServers"][server_name] = {
            "transport": "stdio",
            "command": "python",
            "args": [str(Path(server_path) / "server.py")]
        }
        
        self._save_config(config)
        logger.info(f"Added {server_name} to MCP server configuration")
    
    def stop_mcp_server(self, server_name: str) -> bool:
        """Stop a running MCP server"""
        if server_name not in self.running_servers:
            return False
        
        try:
            process = self.running_servers[server_name]
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            del self.running_servers[server_name]
            if server_name in self.server_ports:
                del self.server_ports[server_name]
            
            logger.info(f"Stopped MCP server: {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop MCP server {server_name}: {e}")
            return False
    
    def stop_all_servers(self):
        """Stop all running MCP servers"""
        for server_name in list(self.running_servers.keys()):
            self.stop_mcp_server(server_name)
    
    def get_running_servers(self) -> Dict[str, int]:
        """Get list of running servers and their ports"""
        return self.server_ports.copy()
    
    def cleanup_server_files(self, server_name: str) -> bool:
        """Remove server files from disk"""
        try:
            server_dir = self.servers_dir / server_name
            if server_dir.exists():
                shutil.rmtree(server_dir)
                logger.info(f"Cleaned up server files: {server_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cleanup server files for {server_name}: {e}")
            return False 