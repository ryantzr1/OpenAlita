"""
MCP Tool Registry - Manages persistent tools following MCP protocol
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excluding function)"""
        data = asdict(self)
        data['function'] = None  # Don't serialize the function
        data['created_at'] = self.created_at.isoformat()
        data['last_used'] = self.last_used.isoformat() if self.last_used else None
        return data

class MCPRegistry:
    """Registry for managing MCP tools with persistence and protocol support"""
    
    def __init__(self, registry_file: str = "mcp_tools_registry.json"):
        self.registry_file = registry_file
        self.tools: Dict[str, MCPTool] = {}
        self.load_registry()
        logger.info(f"MCP Registry initialized with {len(self.tools)} tools")
    
    def register_tool(self, name: str, function: Callable, metadata: Dict[str, Any], script_content: str) -> bool:
        """Register a new MCP tool"""
        try:
            if name in self.tools:
                logger.warning(f"Tool '{name}' already exists, updating...")
            
            tool = MCPTool(
                name=name,
                description=metadata.get('description', ''),
                function=function,
                metadata=metadata,
                script_content=script_content,
                created_at=datetime.now()
            )
            
            self.tools[name] = tool
            self.save_registry()
            logger.info(f"Registered tool: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register tool {name}: {e}")
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
            result = tool.function(*args, **kwargs)
            logger.info(f"Executed tool '{name}' successfully")
            return result
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
        """Load tools from registry file"""
        if not os.path.exists(self.registry_file):
            logger.info("No existing registry file found")
            return
        
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
            
            # Note: Functions can't be deserialized, so we only load metadata
            # Functions need to be recreated from script_content
            logger.info(f"Loaded {len(data)} tool metadata entries from registry")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def save_registry(self):
        """Save tools metadata to registry file"""
        try:
            data = [tool.to_dict() for tool in self.tools.values()]
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(data)} tools to registry")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
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