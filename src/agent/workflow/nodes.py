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
from ..mcp_agent import MCPAgent
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
    
    # Check if we have image files - but don't automatically route to synthesizer
    # Use absolute path to gaia_files directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    gaia_files_dir = os.path.join(project_root, "gaia_files")
    image_files = []
    if os.path.exists(gaia_files_dir):
        image_files = [f for f in os.listdir(gaia_files_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
        if image_files:
            logger.info(f"Found image files: {image_files}")
    
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
    
    # Check for image files
    has_image_files = len(image_files) > 0
    image_files_info = f"Images available: {', '.join(image_files)}" if image_files else "No image files detected"
    
    # Create analysis prompt for LLM
    analysis_prompt = COORDINATOR_ANALYSIS_PROMPT.format(
        query=query,
        context_summary=context_summary,
        has_image_files=has_image_files,
        image_files_info=image_files_info,
        web_results_summary=json.dumps(web_results[:2], indent=2) if web_results else "No web results yet",
        mcp_results_summary=chr(10).join(mcp_results[:2]) if mcp_results else "No tool results yet"
    )
    
    try:
        # Get LLM analysis
        analysis_response = ""
        for chunk in llm_provider._make_api_call(analysis_prompt):
            analysis_response += chunk
        
        # Parse the JSON response
        try:
            json_start = analysis_response.find('{')
            json_end = analysis_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = analysis_response[json_start:json_end]
                analysis = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse coordinator analysis: {e}")
            # Fallback to simple keyword-based routing
            analysis = {
                "next_action": "web_search",
                "reasoning": "Fallback: LLM analysis failed, using web search",
                "confidence": 0.5,
                "search_strategy": "broader"
            }
        
        next_action = analysis.get("next_action", "web_search")
        reasoning = analysis.get("reasoning", "No reasoning provided")
        confidence = analysis.get("confidence", 0.5)
        missing_info = analysis.get("missing_info", "")
        search_strategy = analysis.get("search_strategy", "broader")
        
        # Build streaming chunks
        chunks = [
            f"🧠 **Coordinator:** {reasoning}",
            f"📊 **Confidence:** {confidence:.1%}",
            f"🔄 **Iteration:** {iteration + 1}/{max_iterations}"
        ]
        
        if missing_info:
            chunks.append(f"🔍 **Missing:** {missing_info}")
        
        # Route based on LLM analysis
        if next_action == "browser_automation":
            chunks.append("🌐 **Decision:** Browser automation needed")
            return Command(
                update={
                    "coordinator_analysis": {
                        "reasoning": reasoning,
                        "next_action": "browser_automation",
                        "confidence": confidence,
                        "browser_capabilities_needed": analysis.get("browser_capabilities_needed", []),
                        "requires_visual_analysis": analysis.get("requires_visual_analysis", False),
                        "requires_interaction": analysis.get("requires_interaction", False),
                        "requires_authentication": analysis.get("requires_authentication", False)
                    },
                    "streaming_chunks": chunks + ["→ Routing to browser agent"]
                },
                goto="browser_agent"
            )
        
        elif next_action == "web_search":
            chunks.append("🌐 **Decision:** Web search needed")
            return Command(
                update={
                    "coordinator_analysis": {
                        "reasoning": reasoning,
                        "next_action": "web_search",
                        "confidence": confidence,
                        "search_strategy": search_strategy,
                        "missing_info": missing_info
                    },
                    "streaming_chunks": chunks + ["→ Routing to web agent"]
                },
                goto="web_agent"
            )
        
        elif next_action == "create_tools":
            chunks.append("🛠️ **Decision:** Tool creation needed")
            return Command(
                update={
                    "coordinator_analysis": {
                        "reasoning": reasoning,
                        "next_action": "create_tools",
                        "confidence": confidence
                    },
                    "streaming_chunks": chunks + ["→ Routing to MCP agent"]
                },
                goto="mcp_agent"
            )
        
        else:  # synthesize
            chunks.append("✅ **Decision:** Ready to synthesize")
            return Command(
                update={
                    "coordinator_analysis": {
                        "reasoning": reasoning,
                        "next_action": "synthesize",
                        "confidence": confidence,
                        "local_image_files": analysis.get("local_image_files", False),
                        "vision_analysis_needed": analysis.get("vision_analysis_needed", False)
                    },
                    "streaming_chunks": chunks + ["→ Routing to synthesizer"]
                },
                goto="synthesizer"
            )
        
    except Exception as e:
        logger.error(f"Coordinator LLM analysis error: {e}")
        # Fallback to simple keyword-based routing
        query_lower = query.lower()
        
        # Browser automation keywords
        browser_keywords = ['youtube', 'video', 'watch', 'login', 'click', 'screenshot', 'social media', 'shopping', 'cart', 'navigate', 'go to']
        if any(keyword in query_lower for keyword in browser_keywords):
            return Command(
                update={
                    "coordinator_analysis": {
                        "reasoning": "Fallback: Query contains browser automation keywords",
                        "next_action": "browser_automation"
                    },
                    "streaming_chunks": [f"🌐 **Coordinator (Fallback):** Browser automation needed for: {query}", "→ Routing to browser agent"]
                },
                goto="browser_agent"
            )
        
        # Web search keywords
        web_keywords = ['weather', 'news', 'latest', 'what is', 'how to', 'find', 'search', 'look up']
        if any(keyword in query_lower for keyword in web_keywords):
            return Command(
                update={
                    "coordinator_analysis": {
                        "reasoning": "Fallback: Query needs current information from web",
                        "next_action": "web_search",
                        "search_strategy": "broader"
                    },
                    "streaming_chunks": [f"🌐 **Coordinator (Fallback):** Web search needed for: {query}", "→ Routing to web agent"]
                },
                goto="web_agent"
            )
        
        # Default to synthesizer if we have any results, otherwise try web search
        if web_results or mcp_results:
            return Command(
                update={
                    "coordinator_analysis": {
                        "reasoning": "Fallback: Default route with existing results",
                        "next_action": "synthesize"
                    },
                    "streaming_chunks": [f"🧠 **Coordinator (Fallback):** Default route for: {query}", "→ Routing to synthesizer"]
                },
                goto="synthesizer"
            )
        else:
            return Command(
                update={
                    "coordinator_analysis": {
                        "reasoning": "Fallback: No results yet, trying web search",
                        "next_action": "web_search",
                        "search_strategy": "broader"
                    },
                    "streaming_chunks": [f"🌐 **Coordinator (Fallback):** No results yet for: {query}", "→ Routing to web agent"]
                },
                goto="web_agent"
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

        # Get search query breakdown
        decomp_response = ""
        for chunk in llm_provider._make_api_call(decomposition_prompt):
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
    
    query = state["original_query"]
    
    try:
        # Use the new MCPAgent class
        mcp_agent = MCPAgent()
        chunks, execution_results = mcp_agent.analyze_and_execute(query)
        
        return Command(
            update={
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
    
    # Check if browser automation was successful
    logger.info(f"Evaluator checking browser results: {mcp_results}")
    browser_success = any(
        "browser automation completed" in str(result) or 
        "Successfully searched for and accessed" in str(result) or
        "Successfully searched for and played" in str(result) or
        "Task completed successfully" in str(result)
        for result in mcp_results
    )
    browser_failed = any("browser automation failed" in str(result) for result in mcp_results)
    
    logger.info(f"Browser success detected: {browser_success}")
    logger.info(f"Browser failed detected: {browser_failed}")
    
    # If browser automation was successful, we have enough information
    if browser_success:
        chunks = [f"📊 **Evaluator:** Browser automation completed successfully!\n"]
        chunks.append(f"📈 **Completeness:** 95.0%\n")
        chunks.append("✅ **Decision:** Browser automation provided the required information\n")
        
        return Command(
            update={
                "answer_completeness": 0.95,
                "confidence_score": 0.95,
                "streaming_chunks": chunks
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
        mcp_results_count=len(mcp_results),
        mcp_results_summary=mcp_results_summary,
        iteration=iteration,
        max_iter=max_iter
    )
    
    try:
        # Get LLM evaluation
        eval_response = ""
        for chunk in llm_provider._make_api_call(evaluation_prompt):
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
            # Fallback evaluation
            completeness = min(1.0, (len(web_results) * 0.3 + len(mcp_results) * 0.4 + 0.3))
            evaluation = {
                "completeness_score": completeness,
                "has_sufficient_info": completeness >= 0.7 or iteration >= max_iter,
                "recommendation": "synthesize" if completeness >= 0.7 or iteration >= max_iter else "continue_search",
                "reasoning": "Fallback evaluation based on simple metrics"
            }
        
        completeness = evaluation.get("completeness_score", 0.5)
        should_synthesize = evaluation.get("has_sufficient_info", False) or iteration >= max_iter
        reasoning = evaluation.get("reasoning", "No reasoning provided")
        
        chunks = [f"📊 **Evaluator:** {reasoning}\n"]
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
        
        # Only update streaming_chunks if going to synthesizer (final decision)
        # If going to coordinator, let the next node handle streaming_chunks
        update_dict = {
            "answer_completeness": completeness,
            "confidence_score": completeness
        }
        
        if goto == "synthesizer":
            update_dict["streaming_chunks"] = chunks
        
        return Command(
            update=update_dict,
            goto=goto
        )
        
    except Exception as e:
        logger.error(f"Evaluator error: {e}")
        # Fallback to synthesizer if something goes wrong
        return Command(
            update={
                "streaming_chunks": [f"📊 **Evaluator Error:** {str(e)}\n→ Proceeding to synthesis\n"],
                "answer_completeness": 0.5,
                "confidence_score": 0.5
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
    
    # Check if we have image files
    # Use absolute path to gaia_files directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    gaia_files_dir = os.path.join(project_root, "gaia_files")
    image_files = []
    if os.path.exists(gaia_files_dir):
        image_files = [f for f in os.listdir(gaia_files_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
        logger.info(f"Found {len(image_files)} image files in {gaia_files_dir}: {image_files}")
    else:
        logger.warning(f"gaia_files directory not found: {gaia_files_dir}")
    
    # Create synthesis prompt
    web_results_summary = json.dumps(web_results[:3], indent=2) if web_results else "No web results"
    mcp_results_summary = chr(10).join(mcp_results) if mcp_results else "No tool results"
    
    # Simple prompt - Claude will naturally handle images if they're in the context
    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        web_results_summary=web_results_summary,
        mcp_results_summary=mcp_results_summary
    )
    
    # Add image context if images exist
    if image_files:
        prompt += f"\n\nIMAGES AVAILABLE: {', '.join(image_files)} in gaia_files directory. Please analyze these images to answer the query."
    
    try:
        # Generate final answer
        chunks = ["🎨 **Synthesizer:** Creating final answer...\n"]
        if image_files:
            chunks.append(f"🖼️ **Images detected:** {', '.join(image_files)}\n")
            chunks.append("👁️ **Using Claude's vision capabilities...**\n")
            logger.info(f"Using vision analysis for {len(image_files)} images")
        
        response_chunks = []
        
        # Use vision-enabled API call if images are present
        if image_files:
            # Convert relative paths to absolute paths
            image_paths = [os.path.join(gaia_files_dir, img) for img in image_files]
            logger.info(f"Image paths for vision analysis: {image_paths}")
            
            for chunk in llm_provider._make_vision_api_call(prompt, image_paths):
                response_chunks.append(chunk)
                chunks.append(chunk)
        else:
            # Use regular API call for text-only
            logger.info("Using text-only API call (no images)")
            for chunk in llm_provider._make_api_call(prompt):
                response_chunks.append(chunk)
                chunks.append(chunk)
        
        final_answer = "".join(response_chunks)
        logger.info(f"Generated final answer with {len(response_chunks)} chunks")
        
        return {
            "final_answer": final_answer,
            "confidence_score": 0.9 if image_files else 0.8,
            "streaming_chunks": chunks
        }
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return {
            "final_answer": f"Error generating response: {str(e)}",
            "confidence_score": 0.3,
            "streaming_chunks": [f"❌ **Synthesis error:** {str(e)}\n"]
        } 