"""
MCP Tool Registry - Manages persistent tools following MCP protocol
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import inspect
import shutil

logger = logging.getLogger('alita.mcp_registry')

@dataclass
class MCPTool:
    """Represents a registered MCP tool"""
    name: str
    description: str
    function: Callable
    metadata: Dict[str, Any]
    script_content: str
    created_at: datetime
    usage_count: int = 0
    last_used: Optional[datetime] = None
    file_path: Optional[str] = None  # Path to the tool's Python file
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excluding function)"""
        data = asdict(self)
        data['function'] = None  # Don't serialize the function
        data['created_at'] = self.created_at.isoformat()
        data['last_used'] = self.last_used.isoformat() if self.last_used else None
        return data

class MCPRegistry:
    """Registry for managing MCP tools with persistence and protocol support"""
    
    def __init__(self, registry_file: str = "mcp_tools_registry.json", tools_dir: str = "mcp_tools"):
        self.registry_file = registry_file
        self.tools_dir = tools_dir
        self.tools: Dict[str, MCPTool] = {}
        
        # Ensure tools directory exists
        os.makedirs(self.tools_dir, exist_ok=True)
        
        self.load_registry()
        logger.info(f"MCP Registry initialized with {len(self.tools)} tools")
    
    def _get_tool_file_path(self, tool_name: str) -> str:
        """Get the file path for a tool"""
        # Sanitize tool name for filename
        safe_name = "".join(c for c in tool_name if c.isalnum() or c in ('_', '-')).rstrip()
        return os.path.join(self.tools_dir, f"{safe_name}.py")
    
    def _save_tool_to_file(self, tool_name: str, script_content: str) -> str:
        """Save tool script to a Python file"""
        file_path = self._get_tool_file_path(tool_name)
        
        try:
            # Add header comment with metadata
            header = f"""# MCP Tool: {tool_name}
# Generated: {datetime.now().isoformat()}
# Source: File-based tool generation
# 
# This file contains the implementation of the '{tool_name}' MCP tool.
# The tool is automatically generated and managed by the Alita MCP Registry.
#

"""
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(header + script_content)
            
            logger.info(f"Saved tool '{tool_name}' to file: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save tool '{tool_name}' to file: {e}")
            return None
    
    def _load_tool_from_file(self, tool_name: str) -> Optional[str]:
        """Load tool script from a Python file"""
        file_path = self._get_tool_file_path(tool_name)
        
        if not os.path.exists(file_path):
            logger.warning(f"Tool file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove header comments (lines starting with #)
            lines = content.split('\n')
            script_lines = []
            for line in lines:
                if not line.strip().startswith('#') or not line.strip():
                    script_lines.append(line)
            
            script_content = '\n'.join(script_lines).strip()
            logger.info(f"Loaded tool '{tool_name}' from file: {file_path}")
            return script_content
            
        except Exception as e:
            logger.error(f"Failed to load tool '{tool_name}' from file: {e}")
            return None
    
    def register_tool(self, name: str, function: Callable, metadata: Dict[str, Any], script_content: str) -> bool:
        """Register a new MCP tool with file-based storage"""
        try:
            if name in self.tools:
                logger.warning(f"Tool '{name}' already exists, updating...")
            
            # Save tool to file
            file_path = self._save_tool_to_file(name, script_content)
            
            tool = MCPTool(
                name=name,
                description=metadata.get('description', ''),
                function=function,
                metadata=metadata,
                script_content=script_content,
                created_at=datetime.now(),
                file_path=file_path
            )
            
            self.tools[name] = tool
            logger.info(f"Added tool '{name}' to memory registry (total: {len(self.tools)})")
            
            # Save to registry file
            self.save_registry()
            
            logger.info(f"Registered tool: {name}")
            if file_path:
                logger.info(f"Tool file saved: {file_path}")
            logger.info(f"Tool code for {name}:\n{script_content}\n{'-'*40}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register tool {name}: {e}", exc_info=True)
            return False
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name"""
        tool = self.tools.get(name)
        if tool:
            tool.usage_count += 1
            tool.last_used = datetime.now()
            self.save_registry()
        return tool
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def search_tools(self, query: str) -> List[MCPTool]:
        """Search tools by name, description, or metadata"""
        query_lower = query.lower()
        matching_tools = []
        
        for tool in self.tools.values():
            # Search in name, description, and metadata
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower() or
                any(query_lower in str(v).lower() for v in tool.metadata.values())):
                matching_tools.append(tool)
        
        return matching_tools
    
    def execute_tool(self, name: str, *args, **kwargs) -> Any:
        """Execute a tool by name"""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        try:
            # Check if the function accepts parameters
            sig = inspect.signature(tool.function)
            
            # If the function has no parameters, call without arguments
            if len(sig.parameters) == 0:
                result = tool.function()
                logger.info(f"Executed tool '{name}' successfully (no args)")
                return result
            else:
                # Function accepts parameters, pass them through
                result = tool.function(*args, **kwargs)
                logger.info(f"Executed tool '{name}' successfully (with args)")
                return result
                
        except TypeError as e:
            # Handle parameter mismatch errors
            error_str = str(e).lower()
            if "missing" in error_str and "argument" in error_str:
                logger.warning(f"Parameter mismatch for tool '{name}': {e}")
                logger.info(f"Trying to execute tool '{name}' without arguments as fallback")
                
                try:
                    # Try without arguments as fallback
                    result = tool.function()
                    logger.info(f"Executed tool '{name}' successfully (fallback, no args)")
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback execution also failed for tool '{name}': {fallback_error}")
                    raise e  # Re-raise the original error
            else:
                # Re-raise if it's not a parameter mismatch
                raise
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}")
            raise
    
    def create_tool_chain(self, tool_names: List[str]) -> Callable:
        """Create a chain of tools that execute sequentially"""
        def chain_executor(*args, **kwargs):
            results = []
            current_args = args
            current_kwargs = kwargs
            
            for tool_name in tool_names:
                tool = self.get_tool(tool_name)
                if not tool:
                    raise ValueError(f"Tool '{tool_name}' not found in chain")
                
                try:
                    # Execute tool with current arguments
                    result = tool.function(*current_args, **current_kwargs)
                    results.append({
                        'tool': tool_name,
                        'result': result,
                        'success': True
                    })
                    
                    # Update arguments for next tool (can be customized)
                    current_args = (result,)  # Pass result as first argument
                    current_kwargs = {}
                    
                except Exception as e:
                    results.append({
                        'tool': tool_name,
                        'result': str(e),
                        'success': False
                    })
                    raise
            
            return results
        
        return chain_executor
    
    def suggest_tool_chain(self, query: str) -> List[str]:
        """Suggest a chain of tools for a given query"""
        # This could use LLM to analyze query and suggest tool combinations
        # For now, return tools that match the query
        matching_tools = self.search_tools(query)
        return [tool.name for tool in matching_tools[:3]]  # Limit to 3 tools
    
    def load_registry(self):
        """Load tools from registry file and recreate functions from script_content or files."""
        if not os.path.exists(self.registry_file):
            logger.info("No existing registry file found")
            return

        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)

            logger.info(f"Loaded {len(data)} tool metadata entries from registry")

            # Use MCPFactory to recreate functions from script_content
            from .mcp_factory import MCPFactory
            mcp_factory = MCPFactory()

            for tool_data in data:
                name = tool_data.get('name')
                script_content = tool_data.get('script_content', '')
                metadata = tool_data.get('metadata', {})
                file_path = tool_data.get('file_path')

<<<<<<< Updated upstream
                if name and script_content:
                    function, _ = mcp_factory.create_mcp_from_script(name, script_content)
                    if function:
                        tool = MCPTool(
                            name=name,
                            description=tool_data.get('description', ''),
                            function=function,
                            metadata=metadata,
                            script_content=script_content,
                            created_at=datetime.fromisoformat(tool_data.get('created_at', datetime.now().isoformat())),
                            usage_count=tool_data.get('usage_count', 0),
                            last_used=datetime.fromisoformat(tool_data.get('last_used')) if tool_data.get('last_used') else None
                        )
                        self.tools[name] = tool
                        logger.info(f"Loaded and recreated tool: {name}")
=======
                if name:
                    # Try to load from file first, fallback to script_content
                    if file_path and os.path.exists(file_path):
                        logger.info(f"Loading tool '{name}' from file: {file_path}")
                        file_script_content = self._load_tool_from_file(name)
                        if file_script_content:
                            script_content = file_script_content
                        else:
                            logger.warning(f"Failed to load tool '{name}' from file, using registry content")
                    elif script_content:
                        logger.info(f"Loading tool '{name}' from registry content")
>>>>>>> Stashed changes
                    else:
                        logger.warning(f"No script content available for tool: {name}")
                        continue

                    if script_content:
                        function, _, _ = mcp_factory.create_mcp_from_script(name, script_content)
                        if function:
                            tool = MCPTool(
                                name=name,
                                description=tool_data.get('description', ''),
                                function=function,
                                metadata=metadata,
                                script_content=script_content,
                                created_at=datetime.fromisoformat(tool_data.get('created_at', datetime.now().isoformat())),
                                usage_count=tool_data.get('usage_count', 0),
                                last_used=datetime.fromisoformat(tool_data.get('last_used')) if tool_data.get('last_used') else None,
                                file_path=file_path
                            )
                            self.tools[name] = tool
                            logger.info(f"Loaded and recreated tool: {name}")
                        else:
                            logger.warning(f"Failed to recreate function for tool: {name}")
                    else:
                        logger.warning(f"No script content available for tool: {name}")
                else:
                    logger.warning(f"Skipping tool with missing name: {tool_data}")

            logger.info(f"Successfully loaded {len(self.tools)} tools from registry file")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}", exc_info=True)
    
    def save_registry(self):
        """Save tools metadata to registry file"""
        try:
            data = [tool.to_dict() for tool in self.tools.values()]
            logger.info(f"Attempting to save {len(data)} tools to registry file: {self.registry_file}")
            logger.info(f"Tools to save: {[tool.name for tool in self.tools.values()]}")
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Successfully saved {len(data)} tools to registry file")
            
            # Verify the file was written
            if os.path.exists(self.registry_file):
                file_size = os.path.getsize(self.registry_file)
                logger.info(f"Registry file size: {file_size} bytes")
                if file_size == 0:
                    logger.warning("Registry file is empty after save!")
                else:
                    logger.info("Registry file contains data")
            else:
                logger.error("Registry file does not exist after save!")
                
        except Exception as e:
            logger.error(f"Failed to save registry: {e}", exc_info=True)
    
    def delete_tool(self, name: str) -> bool:
        """Delete a tool and its associated file"""
        try:
            tool = self.tools.get(name)
            if not tool:
                logger.warning(f"Tool '{name}' not found for deletion")
                return False
            
            # Delete the tool file if it exists
            if tool.file_path and os.path.exists(tool.file_path):
                os.remove(tool.file_path)
                logger.info(f"Deleted tool file: {tool.file_path}")
            
            # Remove from memory
            del self.tools[name]
            logger.info(f"Removed tool '{name}' from memory")
            
            # Save updated registry
            self.save_registry()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete tool '{name}': {e}")
            return False
    
    def export_tool(self, name: str, export_path: str) -> bool:
        """Export a tool to a specific location"""
        try:
            tool = self.tools.get(name)
            if not tool:
                logger.warning(f"Tool '{name}' not found for export")
                return False
            
            # Copy the tool file to the export location
            if tool.file_path and os.path.exists(tool.file_path):
                shutil.copy2(tool.file_path, export_path)
                logger.info(f"Exported tool '{name}' to: {export_path}")
                return True
            else:
                logger.warning(f"Tool file not found for export: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to export tool '{name}': {e}")
            return False
    
    def import_tool(self, file_path: str) -> bool:
        """Import a tool from a file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"Import file not found: {file_path}")
                return False
            
            # Extract tool name from filename
            tool_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Load the script content
            with open(file_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # Create function using MCPFactory
            from .mcp_factory import MCPFactory
            mcp_factory = MCPFactory()
            function, metadata, cleaned_script = mcp_factory.create_mcp_from_script(tool_name, script_content)
            
            if function:
                # Register the tool
                success = self.register_tool(tool_name, function, metadata, cleaned_script or script_content)
                if success:
                    logger.info(f"Successfully imported tool '{tool_name}' from: {file_path}")
                    return True
                else:
                    logger.error(f"Failed to register imported tool: {tool_name}")
                    return False
            else:
                logger.error(f"Failed to create function from imported file: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to import tool from {file_path}: {e}")
            return False
    
    def list_tool_files(self) -> List[str]:
        """List all tool files in the tools directory"""
        try:
            if not os.path.exists(self.tools_dir):
                return []
            
            tool_files = []
            for filename in os.listdir(self.tools_dir):
                if filename.endswith('.py'):
                    tool_files.append(filename)
            
            return sorted(tool_files)
            
        except Exception as e:
            logger.error(f"Failed to list tool files: {e}")
            return []
    
    def log_registered_tools(self):
        """Log all registered tools for debugging"""
        logger.info(f"=== MCP Registry Status ===")
        logger.info(f"Total tools registered: {len(self.tools)}")
        logger.info(f"Tools directory: {self.tools_dir}")
        
        for name, tool in self.tools.items():
            logger.info(f"Tool: {name}")
            logger.info(f"  Description: {tool.description}")
            logger.info(f"  Created: {tool.created_at}")
            logger.info(f"  Usage count: {tool.usage_count}")
            logger.info(f"  Last used: {tool.last_used}")
            logger.info(f"  File path: {tool.file_path}")
            logger.info(f"  Metadata: {tool.metadata}")
            logger.info(f"  Script preview: {tool.script_content[:200]}...")
            logger.info("---")
    
    def get_tool_capabilities(self) -> Dict[str, Any]:
        """Get MCP protocol capabilities for tools"""
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {
                        "type": "object",
                        "properties": tool.metadata.get('args', {}),
                        "required": []
                    }
                }
                for tool in self.tools.values()
            ]
        }

    def check_registry_status(self):
        """Check the current status of the registry"""
        logger.info(f"=== MCP Registry Status Check ===")
        logger.info(f"Tools in memory: {len(self.tools)}")
        logger.info(f"Registry file: {self.registry_file}")
        logger.info(f"Tools directory: {self.tools_dir}")
        logger.info(f"Registry file exists: {os.path.exists(self.registry_file)}")
        logger.info(f"Tools directory exists: {os.path.exists(self.tools_dir)}")
        
        if os.path.exists(self.registry_file):
            file_size = os.path.getsize(self.registry_file)
            logger.info(f"Registry file size: {file_size} bytes")
            
            if file_size > 0:
                try:
                    with open(self.registry_file, 'r') as f:
                        data = json.load(f)
                    logger.info(f"Registry file contains {len(data)} tool entries")
                except Exception as e:
                    logger.error(f"Error reading registry file: {e}")
            else:
                logger.warning("Registry file is empty")
        
        # List tool files
        tool_files = self.list_tool_files()
        logger.info(f"Tool files in directory: {len(tool_files)}")
        for tool_file in tool_files:
            logger.info(f"  - {tool_file}")
        
        # List tools in memory
        if self.tools:
            logger.info("Tools in memory:")
            for name, tool in self.tools.items():
                logger.info(f"  - {name}: {tool.description}")
                if tool.file_path:
                    logger.info(f"    File: {tool.file_path}")
        else:
            logger.info("No tools in memory")
        
        logger.info("=" * 40) 