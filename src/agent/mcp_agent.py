"""
MCP Agent Module

Handles intelligent tool analysis, creation, and execution.
"""

import json
import logging
import os
from typing import Dict, Any, List, Tuple

from .llm_provider import LLMProvider
from .mcp_factory import MCPFactory
from .mcp_registry import MCPRegistry
from .prompts import TOOL_REQUIREMENTS_ANALYSIS_PROMPT, TOOL_SCRIPT_GENERATION_PROMPT

logger = logging.getLogger('alita.mcp_agent')


def extract_tool_arguments(tool_name: str, query: str, tool_req: Dict[str, Any]) -> List[Any]:
    """Simple argument extraction based on tool description"""
    description = tool_req.get("description", "").lower()
    purpose = tool_req.get("purpose", "").lower()
    
    # If tool mentions processing input, pass the query
    if any(word in description or word in purpose for word in ["text", "input", "query", "process", "parse", "analyze"]):
        return [query]
    
    # If tool is self-contained (calculations, counters), don't pass arguments
    if any(word in description or word in purpose for word in ["calculate", "count", "compute", "generate"]):
        return []
    
    # Default: pass query for most tools
    return [query]


class MCPAgent:
    """Simple MCP agent for tool creation and execution"""
    
    def __init__(self):
        self.mcp_factory = MCPFactory()
        self.mcp_registry = MCPRegistry()
        self.llm_provider = LLMProvider()
    
    def analyze_and_execute(self, query: str) -> Tuple[List[str], List[str]]:
        """Analyze query for tool requirements and execute appropriate tools"""
        logger.info(f"MCP agent analyzing query: {query}")
        
        chunks = ["🛠️ **MCP Agent:** Analyzing query for tool requirements...\n"]
        
        try:
            # Check if we have image files that might need vision analysis
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            gaia_files_dir = os.path.join(project_root, "gaia_files")
            image_files = []
            if os.path.exists(gaia_files_dir):
                image_files = [f for f in os.listdir(gaia_files_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
                if image_files:
                    logger.info(f"MCP agent found image files: {image_files}")
                    chunks.append(f"🖼️ **Images detected:** {', '.join(image_files)}\n")
            
            # Analyze tool requirements
            tool_requirements, execution_strategy, reasoning = self._analyze_tool_requirements(query)
            
            chunks.append(f"📊 **Analysis:** {reasoning}\n")
            chunks.append(f"🔄 **Strategy:** {execution_strategy}\n")
            
            if not tool_requirements:
                chunks.append("❌ **No tools required** - query doesn't need MCP tools\n")
                return chunks, []
            
            chunks.append(f"🔧 **Tools needed:** {len(tool_requirements)}\n")
            
            # Execute tools
            execution_results = []
            
            if execution_strategy == "parallel":
                # Execute tools in parallel (simplified - just sequential for now)
                for i, tool_req in enumerate(tool_requirements, 1):
                    chunks.append(f"⚡ **Tool {i}:** {tool_req['name']}\n")
                    result = self._execute_tool(tool_req, query, image_files)
                    execution_results.append(result)
                    chunks.append(f"   → {result}\n")
            else:
                # Execute tools sequentially
                for i, tool_req in enumerate(tool_requirements, 1):
                    chunks.append(f"🔧 **Tool {i}:** {tool_req['name']}\n")
                    result = self._execute_tool(tool_req, query, image_files)
                    execution_results.append(result)
                    chunks.append(f"   → {result}\n")
            
            chunks.append("✅ **MCP execution completed**\n")
            return chunks, execution_results
            
        except Exception as e:
            logger.error(f"MCP agent error: {e}")
            return [f"❌ **MCP agent error:** {str(e)}"], []
    
    def _analyze_tool_requirements(self, query: str) -> Tuple[List[Dict], str, str]:
        """Analyze query to determine tool requirements"""
        analysis_prompt = TOOL_REQUIREMENTS_ANALYSIS_PROMPT.format(query=query)
        
        analysis_response = ""
        for chunk in self.llm_provider._make_api_call(analysis_prompt):
            analysis_response += chunk
        
        try:
            json_start = analysis_response.find('{')
            json_end = analysis_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = analysis_response[json_start:json_end]
                analysis_data = json.loads(json_str)
                
                tool_requirements = analysis_data.get("tool_requirements", [])
                execution_strategy = analysis_data.get("execution_strategy", "sequential")
                reasoning = analysis_data.get("reasoning", "No reasoning provided")
                
                return tool_requirements, execution_strategy, reasoning
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse tool analysis: {e}")
            # Simple fallback
            fallback_tool = {
                "name": "_".join(query.split()[:3]).lower(),
                "description": f"Tool for: {query}",
                "purpose": "Handle the user query",
                "dependencies": [],
                "execution_order": 1,
                "can_run_parallel": False
            }
            return [fallback_tool], "sequential", "Fallback: Single tool approach"
    
    def _find_existing_tools(self, tool_requirements: List[Dict]) -> List:
        """Find existing tools that match the requirements"""
        existing_tools = []
        for req in tool_requirements:
            matching_tools = self.mcp_registry.search_tools(req["name"])
            if matching_tools:
                existing_tools.extend(matching_tools)
        return existing_tools
    
    def _create_missing_tools(self, tools_to_create: List[Dict], query: str) -> List[str]:
        """Create missing tools and return status chunks"""
        chunks = [f"🆕 **Creating {len(tools_to_create)} new tools...**"]
        
        for req in tools_to_create:
            tool_name = req["name"]
            chunks.append(f"🔧 **Creating:** {tool_name}")
            
            script_prompt = TOOL_SCRIPT_GENERATION_PROMPT.format(
                tool_name=tool_name,
                description=req["description"],
                purpose=req["purpose"],
                query=query,
                specialized_capabilities=req.get('specialized_capabilities', []),
                api_endpoints=req.get('api_endpoints', []),
                external_services=req.get('external_services', [])
            )

            script_content = ""
            for chunk in self.llm_provider._make_api_call(script_prompt):
                script_content += chunk
            
            function, metadata = self.mcp_factory.create_mcp_from_script(tool_name, script_content)
            
            if function:
                success = self.mcp_registry.register_tool(tool_name, function, metadata, script_content)
                if success:
                    chunks.append(f"   ✅ Registered: {tool_name}")
                else:
                    chunks.append(f"   ❌ Registration failed: {tool_name}")
            else:
                chunks.append(f"   ❌ Creation failed: {tool_name}")
        
        return chunks
    
    def _execute_tools(self, tool_requirements: List[Dict], 
                      execution_strategy: str, query: str, chunks: List[str]) -> List[str]:
        """Execute tools according to the specified strategy"""
        execution_results = []
        
        if execution_strategy == "sequential":
            chunks.append("⚡ **Executing tools sequentially...**")
            
            sorted_requirements = sorted(tool_requirements, key=lambda x: x["execution_order"])
            
            for req in sorted_requirements:
                tool_name = req["name"]
                chunks.append(f"🔍 **Executing:** {tool_name}")
                
                try:
                    tool_args = extract_tool_arguments(tool_name, query, req)
                    
                    if tool_args:
                        result = self.mcp_registry.execute_tool(tool_name, *tool_args)
                    else:
                        result = self.mcp_registry.execute_tool(tool_name)
                    
                    result_str = f"✅ {tool_name}: {str(result)[:100]}..."
                    chunks.append(f"   → Success: {str(result)[:50]}...")
                    execution_results.append(result_str)
                        
                except Exception as e:
                    result_str = f"❌ {tool_name}: {str(e)}"
                    chunks.append(f"   → Failed: {str(e)}")
                    execution_results.append(result_str)
        
        return execution_results
    
    def _execute_tool(self, tool_req: Dict, query: str, image_files: List[str] = None) -> str:
        """Execute a single tool with optional vision support"""
        try:
            # Check if this is a vision-related tool
            if image_files and any(keyword in tool_req.get('description', '').lower() 
                                 for keyword in ['vision', 'image', 'ocr', 'fraction', 'mathematical']):
                logger.info(f"Executing vision-enabled tool: {tool_req['name']}")
                
                # Create vision-enabled prompt
                vision_prompt = f"""Execute the following tool: {tool_req['name']}

Tool Description: {tool_req.get('description', '')}
Purpose: {tool_req.get('purpose', '')}

Query: {query}

Available Images: {', '.join(image_files) if image_files else 'None'}

Please analyze the images and execute the tool functionality. Provide a clear, detailed result."""

                # Use vision-enabled API call
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                gaia_files_dir = os.path.join(project_root, "gaia_files")
                image_paths = [os.path.join(gaia_files_dir, img) for img in image_files] if image_files else []
                
                response_chunks = []
                for chunk in self.llm_provider._make_vision_api_call(vision_prompt, image_paths):
                    response_chunks.append(chunk)
                
                return "".join(response_chunks)
            else:
                # Regular tool execution
                logger.info(f"Executing regular tool: {tool_req['name']}")
                
                tool_prompt = f"""Execute the following tool: {tool_req['name']}

Tool Description: {tool_req.get('description', '')}
Purpose: {tool_req.get('purpose', '')}

Query: {query}

Please execute the tool functionality and provide a clear result."""

                response_chunks = []
                for chunk in self.llm_provider._make_api_call(tool_prompt):
                    response_chunks.append(chunk)
                
                return "".join(response_chunks)
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_req['name']}: {e}")
            return f"Error executing {tool_req['name']}: {str(e)}" 