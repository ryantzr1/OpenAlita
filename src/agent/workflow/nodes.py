"""
LangGraph Workflow Node Functions

Contains all the node functions for the LangGraph workflow.
"""

import json
import logging
import time
import os
from typing import Dict, Any, List, Literal, Annotated
from langgraph.types import Command
from dotenv import load_dotenv

from ..llm_provider import LLMProvider
from ..web_agent import WebAgent
from ..mcp_factory import MCPFactory
from ..mcp_registry import MCPRegistry
from ..prompts import (
    COORDINATOR_ANALYSIS_PROMPT,
    TARGETED_SEARCH_PROMPT,
    VERIFICATION_SEARCH_PROMPT,
    BROADER_SEARCH_PROMPT,
    TOOL_REQUIREMENTS_ANALYSIS_PROMPT,
    TOOL_SCRIPT_GENERATION_PROMPT,
    BROWSER_MCP_ANALYSIS_PROMPT,
    BROWSER_TOOL_SCRIPT_PROMPT,
    EVALUATOR_ANALYSIS_PROMPT,
    SYNTHESIS_PROMPT
)
from .state import State

logger = logging.getLogger('alita.langgraph')


def coordinator_node(state: State) -> Command[Literal["web_agent", "mcp_agent", "synthesizer", "browser_agent"]]:
    """Coordinator node that uses LLM to analyze queries and determine strategy"""
    logger.info("Coordinator analyzing query with LLM...")
    
    llm_provider = LLMProvider()
    query = state["original_query"]
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)
    
    # Check if we've reached max iterations
    if iteration >= max_iterations:
        return Command(
            update={
                "coordinator_analysis": {"reasoning": "Max iterations reached", "next_action": "synthesize"},
                "streaming_chunks": [f"🧠 **Coordinator:** Max iterations ({max_iterations}) reached, synthesizing..."]
            },
            goto="synthesizer"
        )
    
    # Get image files from state
    image_files = state.get("image_files", [])
    
    # Build context about what we've already done
    web_results = state.get("web_search_results", [])
    mcp_results = state.get("mcp_execution_results", [])
    previous_analysis = state.get("coordinator_analysis", {})
    
    # Analyze quality and gaps in existing information
    information_analysis = ""
    if web_results:
        high_quality_results = [r for r in web_results if r.get('credibility_score', 0.5) > 0.7]
        information_analysis += f"High-quality web sources: {len(high_quality_results)}/{len(web_results)}\n"
        
        # Check for recent information
        recent_indicators = ['2024', '2023', 'latest', 'recent', 'today', 'current']
        recent_results = [r for r in web_results if any(indicator in r.get('content', '').lower() for indicator in recent_indicators)]
        information_analysis += f"Recent information sources: {len(recent_results)}/{len(web_results)}\n"
    
    if mcp_results:
        information_analysis += f"Computational results: {len(mcp_results)} tools executed\n"
    
    context_summary = f"""
Current iteration: {iteration + 1}/{max_iterations}
Previous actions: {previous_analysis.get('next_action', 'None')}

INFORMATION QUALITY ANALYSIS:
{information_analysis}

SEARCH RESULTS SUMMARY:
{json.dumps([{'title': r.get('title', ''), 'credibility': r.get('credibility_score', 0.5), 'type': r.get('search_type', 'original')} for r in web_results[:3]], indent=2) if web_results else "No web results yet"}
"""
    
    # Create analysis prompt for LLM with browser-use detection
    has_image_files = "Yes" if image_files else "No"
    image_files_info = f"Image files: {', '.join([os.path.basename(f) for f in image_files])}" if image_files else "No image files detected"
    web_results_summary = json.dumps(web_results[:2], indent=2) if web_results else "No web results yet"
    mcp_results_summary = chr(10).join(mcp_results[:2]) if mcp_results else "No tool results yet"
    
    analysis_prompt = COORDINATOR_ANALYSIS_PROMPT.format(
        query=query,
        context_summary=context_summary,
        has_image_files=has_image_files,
        image_files_info=image_files_info,
        web_results_summary=web_results_summary,
        mcp_results_summary=mcp_results_summary
    )
    
    # Add image context if images exist
    if image_files:
        analysis_prompt += f"\n\nIMAGES AVAILABLE: {', '.join([os.path.basename(f) for f in image_files])} in gaia_files directory. Please consider these images when analyzing the query."
    
    try:
        # Get LLM analysis with vision support
        analysis_response = ""
        for chunk in llm_provider._make_api_call(analysis_prompt, image_files):
            analysis_response += chunk
        
        # Parse the JSON response
        try:
            # Extract JSON from the response (in case there's extra text)
            json_start = analysis_response.find('{')
            json_end = analysis_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = analysis_response[json_start:json_end]
                analysis = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM analysis: {e}")
            # Fallback to simple heuristics with browser-use detection
            query_lower = query.lower()
            browser_keywords = ['youtube', 'video', 'watch', 'login', 'click', 'screenshot', 'social media', 'shopping', 'cart']
            if any(keyword in query_lower for keyword in browser_keywords):
                analysis = {"next_action": "browser_automation", "reasoning": "Fallback: Query contains browser automation keywords", "confidence": 0.8}
            elif "weather" in query_lower or "news" in query_lower or "latest" in query_lower:
                analysis = {"next_action": "web_search", "reasoning": "Fallback: Query needs current information", "confidence": 0.6}
            else:
                analysis = {"next_action": "create_tools", "reasoning": "Fallback: Query needs computation", "confidence": 0.6}
        
        next_action = analysis.get("next_action", "synthesize")
        reasoning = analysis.get("reasoning", "No reasoning provided")
        search_strategy = analysis.get("search_strategy", "broader")
        missing_info = analysis.get("missing_info", "")
        browser_capabilities = analysis.get("browser_capabilities_needed", [])
        
        # Map action to goto
        action_map = {
            "browser_automation": "browser_agent",
            "web_search": "web_agent",
            "create_tools": "mcp_agent", 
            "synthesize": "synthesizer"
        }
        goto = action_map.get(next_action, "synthesizer")
        
        # Enhanced logging with strategy info
        chunks = []
        if image_files:
            chunks.append(f"🖼️ **Vision enabled:** {len(image_files)} images detected")
        
        if next_action == "browser_automation":
            chunks.append(f"🌐 **Coordinator:** {reasoning}")
            chunks.append(f"🤖 **Browser Capabilities:** {', '.join(browser_capabilities)}")
            chunks.append(f"→ Routing to browser automation")
        elif next_action == "web_search":
            chunks.append(f"🧠 **Coordinator:** {reasoning}")
            chunks.append(f"🎯 **Strategy:** {search_strategy} search")
            chunks.append(f"💡 **Missing:** {missing_info}")
            chunks.append(f"→ Routing to {goto}")
        else:
            chunks.append(f"🧠 **Coordinator:** {reasoning}")
            chunks.append(f"→ Routing to {goto}")
        
        # Store enhanced analysis for use by other agents
        enhanced_analysis = analysis.copy()
        enhanced_analysis.update({
            "search_strategy": search_strategy,
            "missing_info": missing_info,
            "browser_capabilities_needed": browser_capabilities,
            "iteration": iteration + 1
        })
        
        return Command(
            update={
                "coordinator_analysis": enhanced_analysis,
                "iteration_count": iteration + 1,
                "streaming_chunks": chunks
            },
            goto=goto
        )
        
    except Exception as e:
        logger.error(f"Coordinator error: {e}")
        # Fallback to synthesizer if something goes wrong
        return Command(
            update={
                "streaming_chunks": [f"🧠 **Coordinator Error:** {str(e)}", "→ Falling back to synthesizer"],
                "iteration_count": iteration + 1
            },
            goto="synthesizer"
        )


def web_agent_node(state: State) -> Command[Literal["evaluator"]]:
    """Web agent node with intelligent query decomposition and strategic searching"""
    logger.info("Web agent analyzing and searching...")
    
    web_agent = WebAgent()
    llm_provider = LLMProvider()
    query = state["original_query"]
    coordinator_analysis = state.get("coordinator_analysis", {})
    search_strategy = coordinator_analysis.get("search_strategy", "broader")
    missing_info = coordinator_analysis.get("missing_info", "")
    existing_results = state.get("web_search_results", [])
    
    # Get image files from state instead of detecting them independently
    image_files = state.get("image_files", [])
    
    # Check if this is a fallback from browser agent
    mcp_results = state.get("mcp_execution_results", [])
    is_browser_fallback = any("browser automation failed" in str(result) for result in mcp_results)
    
    try:
        # Create strategy-aware search prompt
        existing_results_summary = json.dumps([{'title': r.get('title', ''), 'type': r.get('search_type', 'original')} for r in existing_results[:3]], indent=2) if existing_results else "No previous results"
        
        if search_strategy == "targeted" and missing_info:
            decomposition_prompt = TARGETED_SEARCH_PROMPT.format(
                query=query,
                missing_info=missing_info,
                existing_results_summary=existing_results_summary
            )

        elif search_strategy == "verification":
            decomposition_prompt = VERIFICATION_SEARCH_PROMPT.format(
                query=query
            )

        else:  # broader strategy
            decomposition_prompt = BROADER_SEARCH_PROMPT.format(
                query=query
            )

        # Add image context if images exist
        if image_files:
            decomposition_prompt += f"\n\nIMAGES AVAILABLE: {', '.join([os.path.basename(f) for f in image_files])} in gaia_files directory. Please consider these images when decomposing the search query."

        # Get search query breakdown with vision support
        decomp_response = ""
        for chunk in llm_provider._make_api_call(decomposition_prompt, image_files):
            decomp_response += chunk
        
        # Parse the JSON response
        search_queries = []
        try:
            json_start = decomp_response.find('{')
            json_end = decomp_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = decomp_response[json_start:json_end]
                decomp_data = json.loads(json_str)
                search_queries = decomp_data.get("search_queries", [])
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse search queries: {e}")
            # Fallback to original query
            search_queries = [query]
        
        # Adapt search approach based on strategy
        chunks = []
        if image_files:
            chunks.append(f"🖼️ **Vision-enabled search:** {len(image_files)} images detected")
        
        if search_strategy == "targeted":
            # For targeted searches, don't include original query again
            final_search_queries = search_queries[:3]
            chunks.extend([
                f"🌐 **Web Agent:** Performing {len(final_search_queries)} TARGETED searches",
                f"🎯 **Focus:** {missing_info}"
            ])
        elif search_strategy == "verification":
            # For verification, use different sources
            final_search_queries = search_queries[:3]
            chunks.extend([
                f"🌐 **Web Agent:** Performing {len(final_search_queries)} VERIFICATION searches",
                f"🔍 **Goal:** Cross-check existing information"
            ])
        else:
            # Broader strategy - include original query
            search_queries = search_queries[:3]  # Limit to 3 focused queries
            final_search_queries = [query] + search_queries
            chunks.append(f"🌐 **Web Agent:** Performing {len(final_search_queries)} searches (1 original + {len(search_queries)} focused)")
        
        # Perform multiple searches
        all_results = []
        for i, search_query in enumerate(final_search_queries, 1):
            if i == 1:
                chunks.append(f"🔍 **Search {i} (Original):** {search_query[:80]}{'...' if len(search_query) > 80 else ''}")
            else:
                chunks.append(f"🔍 **Search {i} (Focused):** {search_query}")
            
            try:
                results = web_agent.search_web(search_query, num_results=2)  # Fewer per query, more queries
                all_results.extend(results)
                chunks.append(f"   → Found {len(results)} results")
            except Exception as search_error:
                logger.error(f"Search error for '{search_query}': {search_error}")
                chunks.append(f"   → Search failed: {str(search_error)}")
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        chunks.append(f"📋 **Total unique results:** {len(unique_results)}")
        
        # Only update streaming_chunks if not a browser fallback
        update_dict = {"web_search_results": unique_results}
        if not is_browser_fallback:
            update_dict["streaming_chunks"] = chunks
        
        return Command(
            update=update_dict,
            goto="evaluator"
        )
        
    except Exception as e:
        logger.error(f"Web agent error: {e}")
        update_dict = {}
        if not is_browser_fallback:
            update_dict["streaming_chunks"] = [f"❌ **Web agent error:** {str(e)}"]
        
        return Command(
            update=update_dict,
            goto="evaluator"
        )


def mcp_agent_node(state: State) -> Command[Literal["evaluator"]]:
    """MCP agent node for intelligent tool analysis, creation, and execution"""
    logger.info("MCP agent analyzing query for tool requirements...")
    
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
        tool_requirements = []
        execution_strategy = "sequential"
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
        
        chunks.extend([
            f"📋 **Analysis:** {reasoning}",
            f"🔧 **Strategy:** {execution_strategy} execution",
            f"🛠️ **Tools needed:** {len(tool_requirements)}"
        ])
        
        # Step 2: Check for existing tools that could help
        existing_tools = []
        for req in tool_requirements:
            matching_tools = mcp_registry.search_tools(req["name"])
            if matching_tools:
                existing_tools.extend(matching_tools)
                chunks.append(f"✅ **Found existing tool:** {req['name']}")
        
        # Step 3: Create missing tools
        tools_to_create = [req for req in tool_requirements if not any(t.name == req["name"] for t in existing_tools)]
        
        if tools_to_create:
            chunks.append(f"🆕 **Creating {len(tools_to_create)} new tools...**")
            
            # Generate scripts for missing tools
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
                    
                    # Register the tool
                    logger.info(f"Attempting to register tool {tool_name}...")
                    success = mcp_registry.register_tool(tool_name, function, metadata, script_content)
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
        
        # Step 4: Execute tools according to strategy
        execution_results = []
        
        if execution_strategy == "sequential":
            chunks.append("⚡ **Executing tools sequentially...**")
            
            # Sort by execution order
            sorted_requirements = sorted(tool_requirements, key=lambda x: x["execution_order"])
            
            for req in sorted_requirements:
                tool_name = req["name"]
                chunks.append(f"🔍 **Executing:** {tool_name}")
                
                try:
                    # Extract arguments for this tool based on the query
                    tool_args = extract_tool_arguments(req, query, llm_provider)
                    
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


def evaluator_node(state: State) -> Command[Literal["coordinator", "synthesizer"]]:
    """Evaluator node for intelligent answer completeness assessment"""
    logger.info("Evaluator assessing completeness with LLM...")
    
    llm_provider = LLMProvider()
    query = state["original_query"]
    web_results = state.get("web_search_results", [])
    mcp_results = state.get("mcp_execution_results", [])
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 5)
    
    # Get image files from state instead of detecting them independently
    image_files = state.get("image_files", [])
    
    # Check if browser automation was successful
    logger.info(f"Evaluator checking browser results: {mcp_results}")
    browser_success = mcp_results
    browser_failed = any("browser automation failed" in str(result) for result in mcp_results)
    
    logger.info(f"Browser success detected: {browser_success}")
    logger.info(f"Browser failed detected: {browser_failed}")
    
    # If browser automation was successful, we have enough information
    if browser_success:
        # Calculate confidence based on browser results quality
        browser_confidence = 0.85  # Base confidence for successful browser automation
        if image_files:
            browser_confidence += 0.05  # Slight boost for vision tasks
        
        chunks = [f"📊 **Evaluator:** Browser automation completed successfully!\n"]
        chunks.append(f"📈 **Completeness:** 95.0%\n")
        chunks.append(f"🎯 **Confidence:** {browser_confidence:.1%}\n")
        chunks.append("✅ **Decision:** Browser automation provided the required information\n")
        
        return Command(
            update={
                "answer_completeness": 0.95,
                "confidence_score": min(browser_confidence, 1.0),
                "mcp_execution_results": mcp_results
            },
            goto="synthesizer"
        )
    
    # Create evaluation prompt for LLM
    web_results_summary = json.dumps(web_results[:3], indent=2) if web_results else "No web search results"
    mcp_results_summary = chr(10).join(mcp_results[:3]) if mcp_results else "No tool execution results"
    
    evaluation_prompt = EVALUATOR_ANALYSIS_PROMPT.format(
        query=query,
        web_results_count=len(web_results),
        web_results_summary=web_results_summary,
        mcp_results_summary=mcp_results_summary,
        mcp_execution_results=mcp_results,
        iteration=iteration,
        max_iter=max_iter
    )
    
    # Add image context if images exist
    if image_files:
        evaluation_prompt += f"\n\nIMAGES AVAILABLE: {', '.join([os.path.basename(f) for f in image_files])} in gaia_files directory. Please consider these images when evaluating completeness."
    
    try:
        # Get LLM evaluation with vision support if images are available
        eval_response = ""
        for chunk in llm_provider._make_api_call(evaluation_prompt, image_files):
            eval_response += chunk
        
        # Parse the JSON response
        try:
            json_start = eval_response.find('{')
            json_end = eval_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = eval_response[json_start:json_end]
                evaluation = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse evaluation: {e}")
            # Fallback evaluation - boost score if images are available
            base_completeness = min(1.0, (len(web_results) * 0.3 + len(mcp_results) * 0.4 + 0.3))
            if image_files:
                base_completeness = min(1.0, base_completeness + 0.2)  # Boost for images
            evaluation = {
                "completeness_score": base_completeness,
                "has_sufficient_info": base_completeness >= 0.7 or iteration >= max_iter,
                "recommendation": "synthesize" if base_completeness >= 0.7 or iteration >= max_iter else "continue_search",
                "reasoning": f"Fallback evaluation based on simple metrics{' with image boost' if image_files else ''}"
            }
        
        completeness = evaluation.get("completeness_score", 0.5)
        should_synthesize = evaluation.get("has_sufficient_info", False) or iteration >= max_iter
        reasoning = evaluation.get("reasoning", "No reasoning provided")
        
        chunks = [f"📊 **Evaluator:** {reasoning}\n"]
        if image_files:
            chunks.append(f"🖼️ **Vision context:** {len(image_files)} images considered\n")
        chunks.append(f"📈 **Completeness:** {completeness:.1%}\n")
        
        # Decide next action
        if should_synthesize:
            goto = "synthesizer"
            chunks.append("✅ **Decision:** Ready to synthesize final answer\n")
        else:
            goto = "coordinator"
            missing = evaluation.get("missing_aspects", [])
            if missing:
                chunks.append(f"🔍 **Missing:** {', '.join(missing[:3])}\n")
            chunks.append("🔄 **Decision:** Need more information\n")
        
        base_confidence = completeness * 0.8
        if image_files:
            base_confidence += 0.1
        if len(web_results) > 0:
            base_confidence += 0.05
        if len(mcp_results) > 0:
            base_confidence += 0.05
        
        # Only update streaming_chunks if going to synthesizer (final decision)
        # If going to coordinator, let the next node handle streaming_chunks
        update_dict = {
            "answer_completeness": completeness,
            "confidence_score": min(base_confidence, 1.0)
        }
        
        if goto == "synthesizer":
            update_dict["streaming_chunks"] = chunks
        
        return Command(
            update=update_dict,
            goto=goto
        )
        
    except Exception as e:
        logger.error(f"Evaluator error: {e}")
        error_confidence = 0.3
        
        return Command(
            update={
                "streaming_chunks": [f"📊 **Evaluator Error:** {str(e)}\n→ Proceeding to synthesis\n"],
                "answer_completeness": 0.5,
                "confidence_score": min(error_confidence, 1.0)
            },
            goto="synthesizer"
        )


def synthesizer_node(state: State):
    """Synthesizer node for final answer generation with natural vision support"""
    logger.info("Synthesizer creating final answer...")
    
    llm_provider = LLMProvider()
    query = state["original_query"]
    web_results = state.get("web_search_results", [])
    mcp_results = state.get("mcp_execution_results", [])
    
    # Get image files from state instead of detecting them independently
    image_files = state.get("image_files", [])
    
    # Create synthesis prompt
    web_results_summary = json.dumps(web_results[:3], indent=2) if web_results else "No web results"
    mcp_results_summary = chr(10).join(mcp_results) if mcp_results else "No tool results"
    
    # Create a more specific prompt when images are present
    if image_files:
        # For image-based tasks, emphasize the specific question requirements
        prompt = f"""IMPORTANT: You have an image to analyze. Please focus on answering this specific question:

QUESTION: {query}

CRITICAL REQUIREMENTS:
- Analyze the provided image carefully
- Answer the question exactly as asked
- Pay attention to specific formatting requirements (e.g., comma-separated lists, no whitespace)
- Follow any ordering requirements mentioned in the question
- Provide the exact format requested

Web Search Results:
{web_results_summary}

Tool Results:
{mcp_results_summary}

Please analyze the image and provide your answer in the exact format requested by the question."""
    else:
        # Use the standard prompt for non-image tasks
        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            web_results_summary=web_results_summary,
            mcp_results_summary=mcp_results_summary
        )
    
    try:
        # Generate final answer with vision support
        chunks = ["🎨 **Synthesizer:** Creating final answer...\n"]
        if image_files:
            chunks.append(f"🖼️ **Images detected:** {', '.join([os.path.basename(f) for f in image_files])}\n")
            chunks.append("👁️ **Using Claude's vision capabilities...**\n")
        
        response_chunks = []
        # Pass image_files to the LLM provider for vision analysis
        for chunk in llm_provider._make_api_call(prompt, image_files):
            response_chunks.append(chunk)
            chunks.append(chunk)
        
        final_answer = "".join(response_chunks)
        
        # Calculate final confidence based on available information and answer quality
        base_confidence = 0.7  # Base confidence for successful synthesis
        if image_files:
            base_confidence += 0.15  # Significant boost for vision tasks
        if len(web_results) > 0:
            base_confidence += 0.1  # Boost for having web search results
        if len(mcp_results) > 0:
            base_confidence += 0.05  # Small boost for having tool results
        
        # Cap at reasonable maximum
        final_confidence = min(base_confidence, 0.95)  # Never claim 100% confidence
        
        return {
            "final_answer": final_answer,
            "confidence_score": final_confidence,
            "streaming_chunks": chunks
        }
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        error_confidence = 0.1
        
        return {
            "final_answer": f"Error generating response: {str(e)}",
            "confidence_score": error_confidence,
            "streaming_chunks": [f"❌ **Synthesis error:** {str(e)}\n"]
        }


def extract_tool_arguments(tool_metadata, user_query, llm_provider):
    """
    Use the LLM to extract arguments for a tool from the user query.
    """
    prompt = f"""
    Tool: {tool_metadata['name']}
    Description: {tool_metadata['description']}
    Expected arguments: {tool_metadata.get('args', 'None')}
    User query: {user_query}
    
    Extract the arguments from the user query and return them as a Python list in the correct order.
    """
    response = llm_provider.simple_completion(prompt)
    try:
        return eval(response)  # Or use ast.literal_eval for safety
    except Exception:
        return []