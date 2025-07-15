"""
Evaluator Node

Handles answer completeness assessment.
"""

import json
import logging
import os
from typing import Dict, Any, Literal, List
from langgraph.types import Command

from ..llm_provider import LLMProvider
from ..prompts import COMPREHENSIVE_EVALUATION_PROMPT
from .context_utils import apply_context_management, summarize_web_results, summarize_mcp_results
from .state import State

logger = logging.getLogger('alita.evaluator')


def evaluator_node(state: State) -> Command[Literal["coordinator", "synthesizer"]]:
    """Evaluator node for intelligent answer completeness assessment"""
    logger.info("Evaluator assessing completeness with LLM...")
    
    # Apply smart context management first
    state = apply_context_management(state)
    
    llm_provider = LLMProvider()
    query = state["original_query"]
    web_results = state.get("web_search_results", [])
    mcp_results = state.get("mcp_execution_results", [])
    browser_results = state.get("browser_results", [])
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 5)
    image_files = state.get("image_files", [])
    previous_analysis = state.get("coordinator_analysis", {})
    
    # Check if we've reached max iterations
    if iteration >= max_iter:
        logger.info("Max iterations reached, proceeding to synthesis")
        return Command(
            update={
                "answer_completeness": 0.8,
                "streaming_chunks": [f"📊 **Evaluator:** Max iterations ({max_iter}) reached, synthesizing...\n"]
            },
            goto="synthesizer"
        )
    
    # Log what we're evaluating with context about previous actions
    previous_action = previous_analysis.get("next_action", "unknown")
    logger.info(f"Evaluator assessing: query='{query[:100]}...', "
               f"web_results={len(web_results)}, mcp_results={len(mcp_results)}, "
               f"browser_results={len(browser_results)}, iteration={iteration}/{max_iter}, "
               f"previous_action={previous_action}")
    
    # Perform comprehensive evaluation of all available information
    evaluation = _comprehensive_evaluation(
        query, web_results, mcp_results, browser_results, 
        iteration, max_iter, llm_provider, image_files, previous_action
    )
    
    completeness = evaluation.get("completeness_score", 0.5)
    should_synthesize = evaluation.get("has_sufficient_info", False)
    reasoning = evaluation.get("reasoning", "No reasoning provided")
    # Handle both field names for backward compatibility
    next_action = evaluation.get("recommended_action") or evaluation.get("recommendation", "synthesize")
    browser_analysis = evaluation.get("browser_analysis", {})
    
    # Build evaluation chunks
    chunks = _build_evaluation_chunks(
        web_results, mcp_results, browser_results, 
        completeness, reasoning, should_synthesize, image_files, previous_action, browser_analysis
    )
    
    # Determine next step with validation
    if should_synthesize:
        goto = "synthesizer"
        chunks.append("✅ **Decision:** Ready to synthesize final answer\n")
        logger.info(f"Evaluator decision: SYNTHESIZE (completeness={completeness:.2f})")
    else:
        goto = "coordinator"
        missing = evaluation.get("missing_aspects", [])
        if missing:
            chunks.append(f"🔍 **Missing:** {', '.join(missing[:3])}\n")
        chunks.append(f"🔄 **Decision:** Need more information ({next_action})\n")
        logger.info(f"Evaluator decision: CONTINUE ({next_action}, completeness={completeness:.2f})")
    
    # Update state with comprehensive evaluation information
    update_dict = {
        "answer_completeness": completeness,
        "streaming_chunks": chunks,  # Always add streaming chunks
        "evaluation_details": {
            "completeness_score": completeness,
            "has_sufficient_info": should_synthesize,
            "reasoning": reasoning,
            "recommended_action": next_action,
            "missing_aspects": evaluation.get("missing_aspects", []),
            "quality_assessment": evaluation.get("quality_assessment", {}),
            "browser_analysis": browser_analysis,
            "iteration": iteration,
            "previous_action": previous_action
        }
    }
    
    return Command(update=update_dict, goto=goto)
    



def _parse_evaluation_response(eval_response: str, image_files: List[str], iteration: int, max_iter: int) -> Dict[str, Any]:
    """Parse the evaluation response."""
    try:
        json_start = eval_response.find('{')
        json_end = eval_response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = eval_response[json_start:json_end]
            return json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse evaluation: {e}")
        # Fallback evaluation - boost score if images are available
        base_completeness = min(1.0, (iteration * 0.2 + 0.3))
        if image_files:
            base_completeness = min(1.0, base_completeness + 0.2)  # Boost for images
        return {
            "completeness_score": base_completeness,
            "has_sufficient_info": base_completeness >= 0.7 or iteration >= max_iter,
            "recommendation": "synthesize" if base_completeness >= 0.7 or iteration >= max_iter else "continue_search",
            "reasoning": f"Fallback evaluation based on simple metrics{' with image boost' if image_files else ''}"
        }


def _comprehensive_evaluation(query: str, web_results: List[Dict], mcp_results: List[str], 
                            browser_results: List[str], iteration: int, max_iter: int,
                            llm_provider: LLMProvider, image_files: List[str], previous_action: str) -> Dict[str, Any]:
    """Perform comprehensive evaluation of all available information."""
    try:
        # Analyze browser results quality
        browser_analysis = _analyze_browser_results(browser_results)
        
        # Create comprehensive evaluation prompt
        web_results_summary = summarize_web_results(web_results[:3], query, llm_provider) if web_results else "No web search results"
        mcp_results_summary = summarize_mcp_results(mcp_results[:3]) if mcp_results else "No tool execution results"
        browser_results_summary = _summarize_browser_results_for_evaluation(browser_results, query)
        
        evaluation_prompt = COMPREHENSIVE_EVALUATION_PROMPT.format(
            query=query,
            web_results_count=len(web_results),
            web_results_summary=web_results_summary,
            mcp_results_count=len(mcp_results),
            mcp_results_summary=mcp_results_summary,
            browser_results_count=len(browser_results),
            browser_results_summary=browser_results_summary,
            browser_has_useful_results=browser_analysis['has_useful_results'],
            browser_has_failures=browser_analysis['has_failures'],
            browser_has_final_results=browser_analysis['has_final_results'],
            browser_has_actions=browser_analysis['has_actions'],
            iteration=iteration,
            max_iter=max_iter,
            previous_action=previous_action
        )

        # Add image context if available
        if image_files:
            evaluation_prompt += f"\n\nIMAGES AVAILABLE: {', '.join([os.path.basename(f) for f in image_files])} in gaia_files directory. Consider these images when evaluating completeness."

        # Get LLM evaluation
        eval_response = ""
        first_chunk = True
        for chunk in llm_provider._make_api_call(evaluation_prompt, image_files):
            if first_chunk and isinstance(chunk, str) and chunk.startswith("Error:"):
                logger.error(f"LLM API error: {chunk}")
                raise RuntimeError(chunk)
            first_chunk = False
            eval_response += chunk
        
        # Parse the response
        evaluation = _parse_evaluation_response(eval_response, image_files, iteration, max_iter)
        
        # Add browser analysis to evaluation
        evaluation.update({
            "browser_analysis": browser_analysis,
            "iteration": iteration,
            "max_iterations": max_iter
        })
        
        # Log evaluation details for debugging
        logger.info(f"Evaluation result: completeness={evaluation.get('completeness_score', 0.5):.2f}, "
                   f"sufficient={evaluation.get('has_sufficient_info', False)}, "
                   f"action={evaluation.get('recommended_action', 'unknown')}")
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Error in comprehensive evaluation: {e}")
        logger.error(f"Query: {query}, Web results: {len(web_results)}, MCP results: {len(mcp_results)}, Browser results: {len(browser_results)}")
        return _fallback_evaluation(query, web_results, mcp_results, browser_results, iteration, max_iter, image_files)


def _analyze_browser_results(browser_results: List[str]) -> Dict[str, Any]:
    """Analyze browser results for quality indicators."""
    if not browser_results:
        return {
            "has_useful_results": False,
            "has_failures": False,
            "has_final_results": False,
            "has_actions": False,
            "failure_reasons": []
        }
    
    # Check for failures
    failure_indicators = [
        "browser automation failed", "browser agent failed", "failed:", 
        "error:", "incomplete", "did not", "could not", "unable", "timeout"
    ]
    has_failures = any(
        any(indicator in str(result).lower() for indicator in failure_indicators)
        for result in browser_results
    )
    
    # Check for useful results
    useful_indicators = ["final result:", "urls visited:", "actions executed:", "sites:"]
    has_useful_results = any(
        any(indicator in str(result).lower() for indicator in useful_indicators)
        for result in browser_results
    )
    
    # Check for specific result types
    has_final_results = any("final result:" in str(result).lower() for result in browser_results)
    has_actions = any("actions executed:" in str(result).lower() for result in browser_results)
    
    return {
        "has_useful_results": has_useful_results,
        "has_failures": has_failures,
        "has_final_results": has_final_results,
        "has_actions": has_actions,
        "failure_reasons": [str(result) for result in browser_results if any(indicator in str(result).lower() for indicator in failure_indicators)]
    }


def _summarize_browser_results_for_evaluation(browser_results: List[str], query: str) -> str:
    """Create a smart summary of browser results for LLM evaluation."""
    if not browser_results:
        return "No browser automation results"
    
    # Extract key information
    final_result = None
    urls_visited = None
    actions_executed = None
    sites_visited = []
    errors = []
    
    for result in browser_results:
        result_lower = result.lower()
        if "final result:" in result_lower:
            final_result = result.replace("Final result:", "").strip()
        elif "urls visited:" in result_lower:
            urls_visited = result
        elif "actions executed:" in result_lower:
            actions_executed = result
        elif "sites:" in result_lower:
            sites_part = result.replace("Sites:", "").strip()
            sites_visited = [site.strip() for site in sites_part.split(",")]
        elif "errors encountered:" in result_lower or "error:" in result_lower:
            errors.append(result)
    
    # Build structured summary
    summary_parts = []
    
    # Most important: Final result
    if final_result:
        # Truncate final result if too long, but keep more than the current 200 char limit
        if len(final_result) > 800:
            final_result = final_result[:800] + "..."
        summary_parts.append(f"FINAL RESULT: {final_result}")
    
    # Context information
    if urls_visited:
        summary_parts.append(f"CONTEXT: {urls_visited}")
    if actions_executed:
        summary_parts.append(f"CONTEXT: {actions_executed}")
    if sites_visited:
        summary_parts.append(f"CONTEXT: Visited sites: {', '.join(sites_visited[:3])}")
    
    # Error information
    if errors:
        summary_parts.append(f"ERRORS: {'; '.join(errors[:2])}")
    
    # Add assessment hints for the LLM
    if final_result:
        summary_parts.append("ASSESSMENT: Browser automation completed with final result")
    elif errors:
        summary_parts.append("ASSESSMENT: Browser automation encountered errors")
    else:
        summary_parts.append("ASSESSMENT: Browser automation completed without clear final result")
    
    return "\n".join(summary_parts)


def _fallback_evaluation(query: str, web_results: List[Dict], mcp_results: List[str], 
                        browser_results: List[str], iteration: int, max_iter: int, 
                        image_files: List[str]) -> Dict[str, Any]:
    """Fallback evaluation when LLM evaluation fails."""
    logger.warning("Using fallback evaluation")
    
    # Simple heuristics
    has_web_info = len(web_results) > 0
    has_tool_info = len(mcp_results) > 0
    has_browser_info = len(browser_results) > 0
    
    # Basic completeness calculation
    base_completeness = 0.3
    if has_web_info:
        base_completeness += 0.2
    if has_tool_info:
        base_completeness += 0.2
    if has_browser_info:
        base_completeness += 0.2
    if image_files:
        base_completeness += 0.1
    
    # Cap based on iteration
    base_completeness = min(base_completeness + (iteration * 0.1), 0.9)
    
    return {
        "completeness_score": base_completeness,
        "has_sufficient_info": base_completeness >= 0.7 or iteration >= max_iter,
        "reasoning": f"Fallback evaluation: web={has_web_info}, tools={has_tool_info}, browser={has_browser_info}, images={bool(image_files)}",
        "missing_aspects": ["Fallback evaluation - specific gaps unknown"],
        "recommended_action": "synthesize" if base_completeness >= 0.7 or iteration >= max_iter else "web_search",
        "quality_assessment": {
            "web_results_quality": 0.5 if has_web_info else 0.0,
            "mcp_results_quality": 0.5 if has_tool_info else 0.0,
            "browser_results_quality": 0.5 if has_browser_info else 0.0,
            "overall_integration": 0.5
        },
        "next_steps": "Proceed with available information"
    }


def _build_evaluation_chunks(web_results: List[Dict], mcp_results: List[str], browser_results: List[str],
                            completeness: float, reasoning: str, should_synthesize: bool, 
                            image_files: List[str], previous_action: str, browser_analysis: Dict[str, Any]) -> List[str]:
    """Build comprehensive evaluation chunks."""
    chunks = [f"📊 **Evaluator:** {reasoning}\n"]
    
    # Previous action context
    if previous_action and previous_action != "unknown":
        chunks.append(f"📋 **Previous Action:** {previous_action}\n")
    
    # Information summary
    info_summary = []
    if web_results:
        info_summary.append(f"🌐 {len(web_results)} web results")
    if mcp_results:
        info_summary.append(f"🛠️ {len(mcp_results)} tool results")
    if browser_results:
        info_summary.append(f"🌐 {len(browser_results)} browser results")
    if image_files:
        info_summary.append(f"🖼️ {len(image_files)} images")
    
    if info_summary:
        chunks.append(f"📋 **Information:** {', '.join(info_summary)}\n")
    
    # Browser status if applicable
    if browser_results:
        if browser_analysis['has_failures']:
            chunks.append(f"⚠️ **Browser Status:** Failed - {', '.join(browser_analysis['failure_reasons'][:2])}\n")
        elif browser_analysis['has_useful_results']:
            chunks.append(f"✅ **Browser Status:** Successful with useful results\n")
        else:
            chunks.append(f"⚠️ **Browser Status:** Completed but no useful results\n")
    
    # Completeness and decision
    chunks.append(f"📈 **Completeness:** {completeness:.1%}\n")
    
    if should_synthesize:
        chunks.append("✅ **Assessment:** Sufficient information available\n")
    else:
        chunks.append("🔄 **Assessment:** Need more information\n")
    
    return chunks 