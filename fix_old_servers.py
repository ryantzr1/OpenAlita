#!/usr/bin/env python3
"""
Script to fix existing MCP server files to use the correct pattern
"""

import os
import re
from pathlib import Path

def fix_server_file(file_path):
    """Fix a single server file to use the correct MCP pattern"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix imports - remove old imports and add correct ones
        old_imports_pattern = r'from mcp\.types import \([\s\S]*?\)'
        new_imports = '''from mcp.types import (
        TextContent,
        Tool,
    )'''
        
        content = re.sub(old_imports_pattern, new_imports, content)
        
        # Fix list_tools function
        old_list_tools_pattern = r'@server\.list_tools\(\)\s*async def handle_list_tools\(\) -> ListToolsResult:[\s\S]*?return ListToolsResult\([\s\S]*?\)'
        new_list_tools = '''@server.list_tools()
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
        ]'''
        
        content = re.sub(old_list_tools_pattern, new_list_tools, content)
        
        # Fix call_tool function
        old_call_tool_pattern = r'@server\.call_tool\(\)\s*async def handle_call_tool\(name: str, arguments: Dict\[str, Any\]\) -> CallToolResult:[\s\S]*?return CallToolResult\([\s\S]*?\)'
        new_call_tool = '''@server.call_tool()
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
            ]'''
        
        content = re.sub(old_call_tool_pattern, new_call_tool, content)
        
        # Fix server.run call
        old_run_pattern = r'async def main\(\):[\s\S]*?await server\.run\([\s\S]*?\)'
        new_run = '''# Create initialization options and run server
    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)'''
        
        content = re.sub(old_run_pattern, new_run, content)
        
        # Remove the main() call at the end if it exists
        content = re.sub(r'\nif __name__ == "__main__":\s*\n\s*asyncio\.run\(main\(\)\)\s*\n', '\n', content)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Fixed {file_path}")
        return True
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all MCP server files"""
    mcp_servers_dir = Path("mcp_servers")
    
    if not mcp_servers_dir.exists():
        print("mcp_servers directory not found")
        return
    
    fixed_count = 0
    total_count = 0
    
    for server_file in mcp_servers_dir.rglob("server.py"):
        total_count += 1
        if fix_server_file(server_file):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} out of {total_count} server files")

if __name__ == "__main__":
    main() 