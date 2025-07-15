"""
MCP Agent Node

Handles tool creation and execution.
"""

import json
import logging
import os
from typing import Dict, Any, Literal, List
from langgraph.types import Command

from ..llm_provider import LLMProvider
from ..mcp_factory import MCPFactory
from ..mcp_registry import MCPRegistry
from ..prompts import TOOL_REQUIREMENTS_ANALYSIS_PROMPT, TOOL_SCRIPT_GENERATION_PROMPT
from .context_utils import apply_context_management, summarize_mcp_results
from .state import State

logger = logging.getLogger('alita.mcp_agent')


def mcp_agent_node(state: State) -> Command[Literal["evaluator"]]:
    """MCP agent node for intelligent tool analysis, creation, and execution"""
    logger.info("MCP agent analyzing query for tool requirements...")
    
    # Apply smart context management first
    state = apply_context_management(state)
    
    mcp_factory = MCPFactory()
    mcp_registry = MCPRegistry()
    llm_provider = LLMProvider()
    query = state["original_query"]
    coordinator_analysis = state.get("coordinator_analysis", {})
    
    # Get image files from state instead of detecting them independently
    image_files = state.get("image_files", [])
    
    try:
        chunks = ["🛠️ **MCP Agent:** Analyzing query for tool requirements"]
        
        if image_files:
            chunks.append(f"🖼️ **Vision-enabled analysis:** {len(image_files)} images detected")
        
        # Step 1: Analyze the query to determine what tools are needed
        analysis_prompt = TOOL_REQUIREMENTS_ANALYSIS_PROMPT.format(query=query)

        # Add image context if images exist
        if image_files:
            analysis_prompt += f"\n\nIMAGES AVAILABLE: {', '.join([os.path.basename(f) for f in image_files])} in gaia_files directory. Please consider these images when analyzing tool requirements."

        # Get tool requirements analysis with vision support
        analysis_response = ""
        for chunk in llm_provider._make_api_call(analysis_prompt, image_files):
            analysis_response += chunk
        
        # Parse the analysis
        tool_requirements, execution_strategy, reasoning = _parse_tool_analysis(analysis_response, query)
        
        chunks.extend([
            f"📋 **Analysis:** {reasoning}",
            f"🔧 **Strategy:** {execution_strategy} execution",
            f"🛠️ **Tools needed:** {len(tool_requirements)}"
        ])
        
        # Step 2: Check for existing tools that could help
        existing_tools = _find_existing_tools(tool_requirements, mcp_registry, chunks)
        
        # Step 3: Create missing tools
        tools_to_create = [req for req in tool_requirements if not any(t.name == req["name"] for t in existing_tools)]
        
        if tools_to_create:
            chunks.append(f"🆕 **Creating {len(tools_to_create)} new tools...**")
            _create_missing_tools(tools_to_create, query, image_files, llm_provider, mcp_factory, mcp_registry, chunks)
        
        # Step 4: Execute tools according to strategy
        execution_results = _execute_tools(tool_requirements, query, mcp_registry, llm_provider, chunks)
        
        chunks.append(f"📊 **Registry status:** {len(mcp_registry.tools)} total tools available")
        
        # Debug: Check registry status
        logger.info("=== DEBUG: Checking registry status ===")
        mcp_registry.check_registry_status()
        
        return Command(
            update={
                "mcp_tools_created": tool_requirements,
                "mcp_execution_results": execution_results,
                "streaming_chunks": chunks
            },
            goto="evaluator"
        )
        
    except Exception as e:
        logger.error(f"MCP agent error: {e}")
        chunks = [f"❌ **MCP agent error:** {str(e)}"]
        
        return Command(
            update={"streaming_chunks": chunks},
            goto="evaluator"
        )


def _parse_tool_analysis(analysis_response: str, query: str) -> tuple:
    """Parse tool requirements analysis."""
    tool_requirements = []
    execution_strategy = "sequential"
    reasoning = "No reasoning provided"
    
    try:
        json_start = analysis_response.find('{')
        json_end = analysis_response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = analysis_response[json_start:json_end]
            analysis_data = json.loads(json_str)
            tool_requirements = analysis_data.get("tool_requirements", [])
            execution_strategy = analysis_data.get("execution_strategy", "sequential")
            reasoning = analysis_data.get("reasoning", "No reasoning provided")
        else:
            raise ValueError("No JSON found in response")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse tool analysis: {e}")
        # Fallback to single tool
        tool_requirements = [{
            "name": "_".join(query.split()[:3]).lower(),
            "description": f"Tool for: {query}",
            "purpose": "Handle the user query",
            "dependencies": [],
            "execution_order": 1,
            "can_run_parallel": False
        }]
        reasoning = "Fallback: Single tool approach"
    
    return tool_requirements, execution_strategy, reasoning


def _find_existing_tools(tool_requirements: List[Dict], mcp_registry: MCPRegistry, chunks: List[str]) -> List:
    """Find existing tools that could help."""
    existing_tools = []
    for req in tool_requirements:
        matching_tools = mcp_registry.search_tools(req["name"])
        if matching_tools:
            existing_tools.extend(matching_tools)
            chunks.append(f"✅ **Found existing tool:** {req['name']}")
    
    return existing_tools


def _create_missing_tools(tools_to_create: List[Dict], query: str, image_files: List[str],
                         llm_provider: LLMProvider, mcp_factory: MCPFactory, 
                         mcp_registry: MCPRegistry, chunks: List[str]) -> None:
    """Create missing tools."""
    for req in tools_to_create:
        tool_name = req["name"]
        description = req["description"]
        purpose = req["purpose"]
        
        chunks.append(f"🔧 **Creating:** {tool_name}")
        
        # Generate script for this specific tool
        script_prompt = TOOL_SCRIPT_GENERATION_PROMPT.format(
            tool_name=tool_name,
            description=description,
            purpose=purpose,
            query=query
        )

        # Add image context if images exist
        if image_files:
            script_prompt += f"\n\nIMAGES AVAILABLE: {', '.join([os.path.basename(f) for f in image_files])} in gaia_files directory. Consider these images when generating the tool script."

        # Generate script content with vision support
        script_content = ""
        for chunk in llm_provider._make_api_call(script_prompt, image_files):
            script_content += chunk
        
        # Log the generated script for debugging
        logger.info(f"Generated script for {tool_name}:")
        logger.info(f"Script content:\n{script_content}")
        logger.info(f"Script content length: {len(script_content)} characters")
        
        # Create the function
        logger.info(f"Attempting to create function for {tool_name}...")
        function, metadata = mcp_factory.create_mcp_from_script(tool_name, script_content)
        
        if function:
            # Log function details
            logger.info(f"Function {tool_name} created successfully")
            logger.info(f"Function type: {type(function)}")
            logger.info(f"Metadata: {metadata}")
            
            # Register the tool with cleaned script content
            logger.info(f"Attempting to register tool {tool_name}...")
            script_to_register = script_content
            logger.debug(f"Registering tool {tool_name} with script length: {len(script_to_register)}")
            success = mcp_registry.register_tool(tool_name, function, metadata, script_to_register)
            if success:
                chunks.append(f"   ✅ Registered: {tool_name}")
                logger.info(f"Tool {tool_name} registered successfully")
                logger.info(f"Total tools in registry: {len(mcp_registry.tools)}")
            else:
                chunks.append(f"   ❌ Registration failed: {tool_name}")
                logger.error(f"Tool {tool_name} registration failed")
        else:
            chunks.append(f"   ❌ Creation failed: {tool_name}")
            logger.error(f"Function {tool_name} creation failed")
            logger.error(f"Script that failed:\n{script_content}")
            logger.error(f"Function returned: {function}")
            logger.error(f"Metadata returned: {metadata}")


def _execute_tools(tool_requirements: List[Dict], query: str, mcp_registry: MCPRegistry,
                  llm_provider: LLMProvider, chunks: List[str]) -> List[str]:
    """Execute tools according to strategy."""
    execution_results = []
    
    # Sort by execution order
    sorted_requirements = sorted(tool_requirements, key=lambda x: x["execution_order"])
    
    chunks.append("⚡ **Executing tools sequentially...**")
    
    for req in sorted_requirements:
        tool_name = req["name"]
        chunks.append(f"🔍 **Executing:** {tool_name}")
        
        try:
            # Extract arguments for this tool based on the query
            tool_args = _extract_tool_arguments(req, query, llm_provider)
            
            # Execute the tool - pass the query as the first argument
            if tool_args:
                # Pass query as first argument, then any extracted args
                result = mcp_registry.execute_tool(tool_name, query, *tool_args)
                execution_results.append(f"✅ {tool_name}: {str(result)[:100]}...")
                chunks.append(f"   → Success: {str(result)[:50]}...")
                logger.info(f"Tool {tool_name} executed with query and args {tool_args}: {result}")
            else:
                # Pass query as the only argument
                result = mcp_registry.execute_tool(tool_name, query)
                execution_results.append(f"✅ {tool_name}: {str(result)[:100]}...")
                chunks.append(f"   → Success: {str(result)[:50]}...")
                logger.info(f"Tool {tool_name} executed with query: {result}")
                
        except Exception as e:
            execution_results.append(f"❌ {tool_name}: {str(e)}")
            chunks.append(f"   → Failed: {str(e)}")
            logger.error(f"Tool {tool_name} execution failed: {e}")
    
    return execution_results


def _extract_tool_arguments(tool_req: Dict[str, Any], query: str, llm_provider: LLMProvider) -> List[str]:
    """Extract tool arguments from query using LLM"""
    try:
        # Simple argument extraction - can be enhanced later
        args_prompt = f"""
        Extract arguments for tool '{tool_req['name']}' from this query: "{query}"
        
        Tool purpose: {tool_req['purpose']}
        
        Return only the arguments as a JSON array, or empty array if no specific arguments found.
        Example: ["arg1", "arg2"] or []
        """
        
        response = ""
        for chunk in llm_provider._make_api_call(args_prompt):
            response += chunk
        
        # Try to parse JSON response
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                args = json.loads(json_str)
                return args if isinstance(args, list) else []
        except (json.JSONDecodeError, ValueError):
            pass
        
        return []
        
    except Exception as e:
        logger.warning(f"Error extracting tool arguments: {e}")
        return [] 