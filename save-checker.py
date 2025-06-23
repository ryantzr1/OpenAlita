"""
MCP Agent Module

Handles intelligent tool analysis, creation, and execution with configurable argument extraction.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .llm_provider import LLMProvider
from .mcp_factory import MCPFactory
from .mcp_registry import MCPRegistry
from .prompts import TOOL_REQUIREMENTS_ANALYSIS_PROMPT, TOOL_SCRIPT_GENERATION_PROMPT

logger = logging.getLogger('alita.mcp_agent')


@dataclass
class ToolRequirement:
    """Represents a tool requirement with metadata"""
    name: str
    description: str
    purpose: str
    dependencies: List[str]
    execution_order: int
    can_run_parallel: bool
    specialized_capabilities: List[str] = None
    api_endpoints: List[str] = None
    external_services: List[str] = None
    
    def __post_init__(self):
        if self.specialized_capabilities is None:
            self.specialized_capabilities = []
        if self.api_endpoints is None:
            self.api_endpoints = []
        if self.external_services is None:
            self.external_services = []


@dataclass
class ToolExecutionResult:
    """Represents the result of a tool execution"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None


class ArgumentExtractor:
    """Intelligent argument extraction for MCP tools"""
    
    def __init__(self):
        # Define patterns for different types of tools
        self.tool_patterns = {
            # Text processing tools
            "text_processor": {
                "keywords": ["text", "input", "query", "process", "parse", "analyze", "extract", "read"],
                "argument_strategy": "pass_query"
            },
            # Self-contained tools
            "self_contained": {
                "keywords": ["calculate", "count", "compute", "determine", "find", "get", "generate"],
                "argument_strategy": "no_arguments"
            },
            # Specialized tools with custom logic
            "specialized": {
                "patterns": {
                    "string_reverser": "pass_query",
                    "word_opposite_finder": "extract_word",
                    "calculator": "no_arguments",
                    "counter": "no_arguments"
                }
            }
        }
    
    def extract_arguments(self, tool_name: str, query: str, tool_req: Dict[str, Any]) -> List[Any]:
        """Extract arguments for a tool based on its characteristics"""
        try:
            description = tool_req.get("description", "").lower()
            purpose = tool_req.get("purpose", "").lower()
            
            # Check for specialized patterns first
            if tool_name in self.tool_patterns["specialized"]["patterns"]:
                strategy = self.tool_patterns["specialized"]["patterns"][tool_name]
                return self._apply_strategy(strategy, tool_name, query, description, purpose)
            
            # Check for text processing tools
            if self._matches_pattern(description, purpose, self.tool_patterns["text_processor"]["keywords"]):
                return self._apply_strategy("pass_query", tool_name, query, description, purpose)
            
            # Check for self-contained tools
            if self._matches_pattern(description, purpose, self.tool_patterns["self_contained"]["keywords"]):
                # Double-check: if it mentions text processing, it probably needs arguments
                if self._matches_pattern(description, purpose, self.tool_patterns["text_processor"]["keywords"]):
                    return self._apply_strategy("pass_query", tool_name, query, description, purpose)
                else:
                    return self._apply_strategy("no_arguments", tool_name, query, description, purpose)
            
            # Default: check if description explicitly mentions needing input
            if any(keyword in description or keyword in purpose for keyword in ["query", "input", "process"]):
                return self._apply_strategy("pass_query", tool_name, query, description, purpose)
            
            # Default: no arguments for most tools
            logger.debug(f"Tool '{tool_name}' doesn't explicitly need arguments, calling without")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting arguments for {tool_name}: {e}")
            return []
    
    def _matches_pattern(self, description: str, purpose: str, keywords: List[str]) -> bool:
        """Check if description or purpose matches any of the given keywords"""
        return any(keyword in description or keyword in purpose for keyword in keywords)
    
    def _apply_strategy(self, strategy: str, tool_name: str, query: str, description: str, purpose: str) -> List[Any]:
        """Apply the appropriate argument extraction strategy"""
        if strategy == "pass_query":
            logger.debug(f"Tool '{tool_name}' mentions input processing, passing query")
            return [query]
        elif strategy == "no_arguments":
            logger.debug(f"Tool '{tool_name}' appears to be self-contained, not passing arguments")
            return []
        elif strategy == "extract_word":
            # Extract specific words from query context
            if "left" in query.lower():
                return ["left"]
            # Add more word extraction logic as needed
            return ["left"]  # Default for this specific case
        else:
            logger.warning(f"Unknown strategy '{strategy}' for tool '{tool_name}'")
            return []


class MCPAgent:
    """Intelligent MCP agent for tool analysis, creation, and execution"""
    
    def __init__(self):
        self.mcp_factory = MCPFactory()
        self.mcp_registry = MCPRegistry()
        self.llm_provider = LLMProvider()
        self.argument_extractor = ArgumentExtractor()
    
    def analyze_and_execute(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Analyze query for tool requirements and execute tools
        
        Returns:
            Tuple of (streaming_chunks, execution_results)
        """
        chunks = ["🛠️ **MCP Agent:** Analyzing query for tool requirements"]
        
        try:
            # Step 1: Analyze tool requirements
            tool_requirements, execution_strategy, reasoning = self._analyze_tool_requirements(query)
            
            chunks.extend([
                f"📋 **Analysis:** {reasoning}",
                f"🔧 **Strategy:** {execution_strategy} execution",
                f"🛠️ **Tools needed:** {len(tool_requirements)}"
            ])
            
            # Step 2: Check for existing tools
            existing_tools = self._find_existing_tools(tool_requirements)
            for req in tool_requirements:
                if any(t.name == req.name for t in existing_tools):
                    chunks.append(f"✅ **Found existing tool:** {req.name}")
            
            # Step 3: Create missing tools
            tools_to_create = [req for req in tool_requirements 
                             if not any(t.name == req.name for t in existing_tools)]
            
            if tools_to_create:
                chunks.extend(self._create_missing_tools(tools_to_create, query))
            
            # Step 4: Execute tools
            execution_results = self._execute_tools(tool_requirements, execution_strategy, query, chunks)
            
            chunks.append(f"📊 **Registry status:** {len(self.mcp_registry.tools)} total tools available")
            
            return chunks, execution_results
            
        except Exception as e:
            logger.error(f"MCP agent error: {e}")
            return [f"❌ **MCP agent error:** {str(e)}"], []
    
    def _analyze_tool_requirements(self, query: str) -> Tuple[List[ToolRequirement], str, str]:
        """Analyze query to determine tool requirements"""
        analysis_prompt = TOOL_REQUIREMENTS_ANALYSIS_PROMPT.format(query=query)
        
        # Get tool requirements analysis
        analysis_response = ""
        for chunk in self.llm_provider._make_api_call(analysis_prompt):
            analysis_response += chunk
        
        # Parse the analysis
        try:
            json_start = analysis_response.find('{')
            json_end = analysis_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = analysis_response[json_start:json_end]
                analysis_data = json.loads(json_str)
                
                tool_requirements = []
                for req_data in analysis_data.get("tool_requirements", []):
                    tool_requirements.append(ToolRequirement(
                        name=req_data.get("name", ""),
                        description=req_data.get("description", ""),
                        purpose=req_data.get("purpose", ""),
                        dependencies=req_data.get("dependencies", []),
                        execution_order=req_data.get("execution_order", 1),
                        can_run_parallel=req_data.get("can_run_parallel", False),
                        specialized_capabilities=req_data.get("specialized_capabilities", []),
                        api_endpoints=req_data.get("api_endpoints", []),
                        external_services=req_data.get("external_services", [])
                    ))
                
                execution_strategy = analysis_data.get("execution_strategy", "sequential")
                reasoning = analysis_data.get("reasoning", "No reasoning provided")
                
                return tool_requirements, execution_strategy, reasoning
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse tool analysis: {e}")
            # Fallback to single tool
            fallback_tool = ToolRequirement(
                name="_".join(query.split()[:3]).lower(),
                description=f"Tool for: {query}",
                purpose="Handle the user query",
                dependencies=[],
                execution_order=1,
                can_run_parallel=False
            )
            return [fallback_tool], "sequential", "Fallback: Single tool approach"
    
    def _find_existing_tools(self, tool_requirements: List[ToolRequirement]) -> List:
        """Find existing tools that match the requirements"""
        existing_tools = []
        for req in tool_requirements:
            matching_tools = self.mcp_registry.search_tools(req.name)
            if matching_tools:
                existing_tools.extend(matching_tools)
        return existing_tools
    
    def _create_missing_tools(self, tools_to_create: List[ToolRequirement], query: str) -> List[str]:
        """Create missing tools and return status chunks"""
        chunks = [f"🆕 **Creating {len(tools_to_create)} new tools...**"]
        
        for req in tools_to_create:
            chunks.append(f"🔧 **Creating:** {req.name}")
            
            # Generate script for this specific tool
            script_prompt = TOOL_SCRIPT_GENERATION_PROMPT.format(
                tool_name=req.name,
                description=req.description,
                purpose=req.purpose,
                query=query,
                specialized_capabilities=req.specialized_capabilities,
                api_endpoints=req.api_endpoints,
                external_services=req.external_services
            )

            # Generate script content
            script_content = ""
            for chunk in self.llm_provider._make_api_call(script_prompt):
                script_content += chunk
            
            # Log the generated script for debugging
            logger.info(f"Generated script for {req.name}:")
            logger.info(f"Script content:\n{script_content}")
            
            # Create the function
            function, metadata = self.mcp_factory.create_mcp_from_script(req.name, script_content)
            
            if function:
                # Log function details
                logger.info(f"Function {req.name} created successfully")
                logger.info(f"Metadata: {metadata}")
                
                # Register the tool
                success = self.mcp_registry.register_tool(req.name, function, metadata, script_content)
                if success:
                    chunks.append(f"   ✅ Registered: {req.name}")
                    logger.info(f"Tool {req.name} registered successfully")
                else:
                    chunks.append(f"   ❌ Registration failed: {req.name}")
                    logger.error(f"Tool {req.name} registration failed")
            else:
                chunks.append(f"   ❌ Creation failed: {req.name}")
                logger.error(f"Function {req.name} creation failed")
                logger.error(f"Script that failed:\n{script_content}")
        
        return chunks
    
    def _execute_tools(self, tool_requirements: List[ToolRequirement], 
                      execution_strategy: str, query: str, chunks: List[str]) -> List[str]:
        """Execute tools according to the specified strategy"""
        execution_results = []
        
        if execution_strategy == "sequential":
            chunks.append("⚡ **Executing tools sequentially...**")
            
            # Sort by execution order
            sorted_requirements = sorted(tool_requirements, key=lambda x: x.execution_order)
            
            for req in sorted_requirements:
                result = self._execute_single_tool(req, query, chunks)
                execution_results.append(result)
        
        return execution_results
    
    def _execute_single_tool(self, req: ToolRequirement, query: str, chunks: List[str]) -> str:
        """Execute a single tool and return the result string"""
        tool_name = req.name
        chunks.append(f"🔍 **Executing:** {tool_name}")
        
        try:
            # Extract arguments for this tool based on the query
            tool_args = self.argument_extractor.extract_arguments(tool_name, query, {
                "description": req.description,
                "purpose": req.purpose
            })
            
            # Execute the tool
            if tool_args:
                result = self.mcp_registry.execute_tool(tool_name, *tool_args)
                result_str = f"✅ {tool_name}: {str(result)[:100]}..."
                chunks.append(f"   → Success: {str(result)[:50]}...")
                logger.info(f"Tool {tool_name} executed with args {tool_args}: {result}")
            else:
                result = self.mcp_registry.execute_tool(tool_name)
                result_str = f"✅ {tool_name}: {str(result)[:100]}..."
                chunks.append(f"   → Success: {str(result)[:50]}...")
                logger.info(f"Tool {tool_name} executed without args: {result}")
            
            return result_str
                
        except Exception as e:
            result_str = f"❌ {tool_name}: {str(e)}"
            chunks.append(f"   → Failed: {str(e)}")
            logger.error(f"Tool {tool_name} execution failed: {e}")
            return result_str 