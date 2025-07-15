"""
Coordinator Node

Handles query analysis and routing decisions.
"""

import json
import logging
import os
from typing import Dict, Any, Literal, List
from langgraph.types import Command

from ..llm_provider import LLMProvider
from ..prompts import COORDINATOR_ANALYSIS_PROMPT
from ..context_manager import context_manager
from .context_utils import apply_context_management, summarize_web_results, summarize_mcp_results
from .state import State

logger = logging.getLogger('alita.coordinator')


def coordinator_node(state: State) -> Command[Literal["web_agent", "mcp_agent", "synthesizer", "browser_agent"]]:
    """Coordinator node that uses LLM to analyze queries and determine strategy"""
    logger.info("Coordinator analyzing query with LLM...")
    
    # Apply smart context management first
    state = apply_context_management(state)
    
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
    browser_results = state.get("browser_results", [])
    previous_analysis = state.get("coordinator_analysis", {})
    evaluation_details = state.get("evaluation_details", {})
    
    # Analyze quality and gaps in existing information
    information_analysis = _build_information_analysis(web_results, mcp_results, browser_results)
    
    # Include evaluation feedback in context
    evaluation_feedback = ""
    if evaluation_details:
        prev_action = evaluation_details.get("previous_action", "unknown")
        completeness = evaluation_details.get("completeness_score", 0.0)
        missing_aspects = evaluation_details.get("missing_aspects", [])
        prev_action_success = evaluation_details.get("previous_action_success", True)
        
        evaluation_feedback = f"""
EVALUATION FEEDBACK:
- Previous action: {prev_action}
- Completeness: {completeness:.1%}
- Previous action successful: {prev_action_success}
- Missing aspects: {', '.join(missing_aspects[:3]) if missing_aspects else 'None identified'}
"""
    
    context_summary = f"""
Current iteration: {iteration + 1}/{max_iterations}
Previous actions: {previous_analysis.get('next_action', 'None')}

INFORMATION QUALITY ANALYSIS:
{information_analysis}

{evaluation_feedback}

SEARCH RESULTS SUMMARY:
{json.dumps([{'title': r.get('title', ''), 'credibility': r.get('credibility_score', 0.5), 'type': r.get('search_type', 'original')} for r in web_results[:3]], indent=2) if web_results else "No web results yet"}
"""
    
    # Create analysis prompt for LLM with browser-use detection
    has_image_files = "Yes" if image_files else "No"
    image_files_info = f"Image files: {', '.join([os.path.basename(f) for f in image_files])}" if image_files else "No image files detected"
    
    # Use enhanced summarization for web results
    web_results_summary = summarize_web_results(web_results[:2], query, llm_provider) if web_results else "No web results yet"
    mcp_results_summary = summarize_mcp_results(mcp_results[:2]) if mcp_results else "No tool results yet"
    browser_results_summary = summarize_mcp_results(browser_results[:2]) if browser_results else "No browser results yet"
    
    analysis_prompt = COORDINATOR_ANALYSIS_PROMPT.format(
        query=query,
        context_summary=context_summary,
        has_image_files=has_image_files,
        image_files_info=image_files_info,
        web_results_summary=web_results_summary,
        mcp_results_summary=mcp_results_summary,
        browser_results_summary=browser_results_summary
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
        analysis = _parse_analysis_response(analysis_response, query)
        
        next_action = analysis.get("next_action", "synthesize")
        reasoning = analysis.get("reasoning", "No reasoning provided")
        search_strategy = analysis.get("search_strategy", "broader")
        missing_info = analysis.get("missing_info", "")
        browser_capabilities = analysis.get("browser_capabilities_needed", [])
        
        # Consider evaluation feedback when making decisions
        if evaluation_details:
            prev_action = evaluation_details.get("previous_action", "")
            prev_action_success = evaluation_details.get("previous_action_success", True)
            
            # If previous action failed, try a different approach
            if not prev_action_success and prev_action == next_action:
                logger.info(f"Previous action {prev_action} failed, trying alternative approach")
                if prev_action == "web_search":
                    next_action = "browser_automation"
                    reasoning += " (Previous web search failed, trying browser automation)"
                elif prev_action == "browser_automation":
                    next_action = "web_search"
                    reasoning += " (Previous browser automation failed, trying web search)"
                elif prev_action == "create_tools":
                    next_action = "web_search"
                    reasoning += " (Previous tool creation failed, trying web search)"
        
        # Map action to goto
        action_map = {
            "browser_automation": "browser_agent",
            "web_search": "web_agent",
            "create_tools": "mcp_agent", 
            "synthesize": "synthesizer"
        }
        goto = action_map.get(next_action, "synthesizer")
        
        # Enhanced logging with strategy info
        chunks = _build_coordinator_chunks(image_files, next_action, reasoning, search_strategy, missing_info, browser_capabilities, goto, evaluation_details)
        
        # Store enhanced analysis for use by other agents
        enhanced_analysis = analysis.copy()
        enhanced_analysis.update({
            "search_strategy": search_strategy,
            "missing_info": missing_info,
            "browser_capabilities_needed": browser_capabilities,
            "iteration": iteration + 1,
            "considered_evaluation_feedback": bool(evaluation_details)
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


def _build_information_analysis(web_results: List[Dict], mcp_results: List[str], browser_results: List[str]) -> str:
    """Build information quality analysis."""
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
    
    if browser_results:
        # Check if browser results are useful
        browser_has_useful_results = any(
            "final result:" in str(result).lower() or
            "urls visited:" in str(result).lower() or
            "actions executed:" in str(result).lower() or
            "sites:" in str(result).lower()
            for result in browser_results
        )
        if browser_has_useful_results:
            information_analysis += f"Browser automation: {len(browser_results)} successful results\n"
        else:
            information_analysis += f"Browser automation: {len(browser_results)} results (may need more info)\n"
    
    return information_analysis


def _parse_analysis_response(analysis_response: str, query: str) -> Dict[str, Any]:
    """Parse the LLM analysis response."""
    try:
        # Extract JSON from the response (in case there's extra text)
        json_start = analysis_response.find('{')
        json_end = analysis_response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = analysis_response[json_start:json_end]
            return json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")
            
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse LLM analysis: {e}")
        # Fallback to simple heuristics with browser-use detection
        return _fallback_analysis(query)


def _fallback_analysis(query: str) -> Dict[str, Any]:
    """Fallback analysis when LLM parsing fails."""
    query_lower = query.lower()
    browser_keywords = ['youtube', 'video', 'watch', 'login', 'click', 'screenshot', 'social media', 'shopping', 'cart']
    if any(keyword in query_lower for keyword in browser_keywords):
        return {"next_action": "browser_automation", "reasoning": "Fallback: Query contains browser automation keywords", "confidence": 0.8}
    elif "weather" in query_lower or "news" in query_lower or "latest" in query_lower:
        return {"next_action": "web_search", "reasoning": "Fallback: Query needs current information", "confidence": 0.6}
    else:
        return {"next_action": "create_tools", "reasoning": "Fallback: Query needs computation", "confidence": 0.6}


def _build_coordinator_chunks(image_files: List[str], next_action: str, reasoning: str, 
                            search_strategy: str, missing_info: str, browser_capabilities: List[str], 
                            goto: str, evaluation_details: Dict[str, Any]) -> List[str]:
    """Build streaming chunks for coordinator output."""
    chunks = []
    
    # Include evaluation feedback if available
    if evaluation_details:
        prev_action = evaluation_details.get("previous_action", "unknown")
        completeness = evaluation_details.get("completeness_score", 0.0)
        prev_action_success = evaluation_details.get("previous_action_success", True)
        
        chunks.append(f"📊 **Previous Evaluation:** {completeness:.1%} completeness")
        if not prev_action_success:
            chunks.append(f"⚠️ **Previous Action Failed:** {prev_action}")
        else:
            chunks.append(f"✅ **Previous Action:** {prev_action}")
    
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
    
    return chunks 