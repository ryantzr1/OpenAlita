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

# Constants
MAX_ITERATIONS = 5
ACTION_MAP = {
    "browser_automation": "browser_agent",
    "web_search": "web_agent", 
    "create_tools": "mcp_agent",
    "synthesize": "synthesizer"
}

def coordinator_node(state: State) -> Command[Literal["web_agent", "mcp_agent", "synthesizer", "browser_agent"]]:
    """Coordinator node that uses LLM to analyze queries and determine strategy"""
    logger.info("Coordinator analyzing query with LLM...")
    
    # Apply smart context management first
    state = apply_context_management(state)
    
    # Check for early termination conditions
    early_termination = _check_early_termination(state)
    if early_termination:
        return early_termination
    
    # Prepare analysis context
    analysis_context = _build_analysis_context(state)
    
    # Get LLM analysis
    try:
        analysis = _get_llm_analysis(state, analysis_context)
        return _build_coordinator_command(state, analysis)
    except Exception as e:
        logger.error(f"Coordinator error: {e}")
        return _build_fallback_command(state, str(e))


def _check_early_termination(state: State) -> Command | None:
    """Check if workflow should terminate early"""
    final_answer = state.get("final_answer")
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", MAX_ITERATIONS)
    
    # Check if synthesizer has already provided a complete answer
    if final_answer and final_answer.strip() and final_answer != "Error generating response: None":
        logger.info(f"Final answer detected: {final_answer[:100]}...")
        logger.info("Synthesizer has already provided a final answer, stopping workflow")
        return Command(
            update={
                "coordinator_analysis": {"reasoning": "Synthesizer already provided complete answer", "next_action": "end"},
                "streaming_chunks": ["🧠 **Coordinator:** Final answer already provided by synthesizer, stopping workflow..."]
            },
            goto="synthesizer"
        )
    
    # Check if we've reached max iterations
    if iteration >= max_iterations:
        return Command(
            update={
                "coordinator_analysis": {"reasoning": "Max iterations reached", "next_action": "synthesize"},
                "streaming_chunks": [f"🧠 **Coordinator:** Max iterations ({max_iterations}) reached, synthesizing..."]
            },
            goto="synthesizer"
        )
    
    return None


def _build_analysis_context(state: State) -> Dict[str, Any]:
    """Build comprehensive context for LLM analysis"""
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", MAX_ITERATIONS)
    
    # Gather existing results
    web_results = state.get("web_search_results", [])
    mcp_results = state.get("mcp_execution_results", [])
    browser_results = state.get("browser_results", [])
    previous_analysis = state.get("coordinator_analysis", {})
    evaluation_details = state.get("evaluation_details", {})
    
    # Track action history
    action_history = state.get("action_history", [])
    if previous_analysis and previous_analysis.get("next_action"):
        action_history.append(previous_analysis.get("next_action"))
    
    # Build context components
    information_analysis = _build_information_analysis(web_results, mcp_results, browser_results)
    evaluation_feedback = _build_evaluation_feedback(evaluation_details)
    action_history_summary = _build_action_history_summary(action_history)
    
    return {
        "iteration": iteration,
        "max_iterations": max_iterations,
        "previous_analysis": previous_analysis,
        "information_analysis": information_analysis,
        "evaluation_feedback": evaluation_feedback,
        "action_history_summary": action_history_summary,
        "action_history": action_history,
        "web_results": web_results,
        "mcp_results": mcp_results,
        "browser_results": browser_results
    }


def _build_evaluation_feedback(evaluation_details: Dict[str, Any]) -> str:
    """Build evaluation feedback string"""
    if not evaluation_details:
        return ""
    
    prev_action = evaluation_details.get("previous_action", "unknown")
    completeness = evaluation_details.get("completeness_score", 0.0)
    missing_aspects = evaluation_details.get("missing_aspects", [])
    prev_action_success = evaluation_details.get("previous_action_success", True)
    
    return f"""
EVALUATION FEEDBACK:
- Previous action: {prev_action}
- Completeness: {completeness:.1%}
- Previous action successful: {prev_action_success}
- Missing aspects: {', '.join(missing_aspects[:3]) if missing_aspects else 'None identified'}
"""


def _build_action_history_summary(action_history: List[str]) -> str:
    """Build action history summary"""
    browser_attempts = action_history.count("browser_automation")
    web_attempts = action_history.count("web_search")
    tool_attempts = action_history.count("create_tools")
    
    return f"""
ACTION HISTORY ANALYSIS:
- Browser automation attempts: {browser_attempts}
- Web search attempts: {web_attempts}
- Tool creation attempts: {tool_attempts}
- Total iterations: {len(action_history)}
"""


def _get_llm_analysis(state: State, context: Dict[str, Any]) -> Dict[str, Any]:
    """Get LLM analysis with enhanced exploration logic"""
    llm_provider = LLMProvider()
    query = state["original_query"]
    image_files = state.get("image_files", [])
    
    # Build analysis prompt
    analysis_prompt = _build_analysis_prompt(query, context, image_files, llm_provider)
    
    # Get LLM response
    analysis_response = ""
    for chunk in llm_provider._make_api_call(analysis_prompt, image_files):
        analysis_response += chunk
    
    # Parse and enhance analysis
    analysis = _parse_analysis_response(analysis_response, query)
    return _apply_exploration_logic(analysis, context)


def _build_analysis_prompt(query: str, context: Dict[str, Any], image_files: List[str], llm_provider: LLMProvider) -> str:
    """Build the analysis prompt for LLM"""
    iteration = context["iteration"]
    max_iterations = context["max_iterations"]
    previous_analysis = context["previous_analysis"]
    information_analysis = context["information_analysis"]
    evaluation_feedback = context["evaluation_feedback"]
    action_history_summary = context["action_history_summary"]
    web_results = context["web_results"]
    mcp_results = context["mcp_results"]
    browser_results = context["browser_results"]
    
    # Build context summary
    context_summary = f"""
Current iteration: {iteration + 1}/{max_iterations}
Previous actions: {previous_analysis.get('next_action', 'None')}

{action_history_summary}

INFORMATION QUALITY ANALYSIS:
{information_analysis}

{evaluation_feedback}

SEARCH RESULTS SUMMARY:
{json.dumps([{'title': r.get('title', ''), 'credibility': r.get('credibility_score', 0.5), 'type': r.get('search_type', 'original')} for r in web_results[:3]], indent=2) if web_results else "No web results yet"}
"""
    
    # Prepare image information
    has_image_files = "Yes" if image_files else "No"
    image_files_info = f"Image files: {', '.join([os.path.basename(f) for f in image_files])}" if image_files else "No image files detected"
    
    # Summarize results
    web_results_summary = summarize_web_results(web_results[:2], query, llm_provider) if web_results else "No web results yet"
    mcp_results_summary = summarize_mcp_results(mcp_results[:2]) if mcp_results else "No tool results yet"
    browser_results_summary = summarize_mcp_results(browser_results[:2]) if browser_results else "No browser results yet"
    
    # Build prompt
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
    
    return analysis_prompt


def _apply_exploration_logic(analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Apply smart exploration logic based on evaluation feedback"""
    evaluation_details = context.get("evaluation_details", {})
    action_history = context["action_history"]
    iteration = context["iteration"]
    
    if not evaluation_details:
        return analysis
    
    prev_action = evaluation_details.get("previous_action", "")
    prev_action_success = evaluation_details.get("previous_action_success", True)
    completeness = evaluation_details.get("completeness_score", 0.0)
    next_action = analysis.get("next_action", "synthesize")
    
    # Exploration logic
    exploration_reason = _determine_exploration_reason(
        prev_action, prev_action_success, completeness, iteration, 
        action_history, next_action
    )
    
    if exploration_reason:
        logger.info(f"Exploring different approach: {exploration_reason}")
        analysis["reasoning"] = analysis.get("reasoning", "") + f" ({exploration_reason})"
        analysis["next_action"] = _get_exploration_action(action_history, completeness)
    
    return analysis


def _determine_exploration_reason(prev_action: str, prev_action_success: bool, 
                                completeness: float, iteration: int, 
                                action_history: List[str], next_action: str) -> str:
    """Determine if exploration is needed and why"""
    browser_attempts = action_history.count("browser_automation")
    web_attempts = action_history.count("web_search")
    
    # Case 1: Previous action explicitly failed
    if not prev_action_success:
        return f"Previous action {prev_action} failed"
    
    # Case 2: Low progress after multiple attempts
    elif completeness < 0.3 and iteration >= 2:
        return f"Low progress ({completeness:.1%}) after {iteration} attempts"
    
    # Case 3: Same action being repeated without improvement
    elif prev_action == next_action and iteration >= 2:
        return f"Repeating {prev_action} without improvement"
    
    # Case 4: Multiple browser attempts without success
    elif browser_attempts >= 2 and completeness < 0.5:
        return f"Multiple browser attempts ({browser_attempts}) with low completeness ({completeness:.1%}), trying web search"
    
    # Case 5: Multiple web search attempts without success
    elif web_attempts >= 2 and completeness < 0.5:
        return f"Multiple web search attempts ({web_attempts}) with low completeness ({completeness:.1%}), trying browser automation"
    
    # Case 6: Alternating between approaches without progress
    elif len(action_history) >= 4 and completeness < 0.4:
        recent_actions = action_history[-4:]
        browser_web_alternating = (
            len(recent_actions) >= 4 and
            any(recent_actions[i] == "browser_automation" and recent_actions[i+1] == "web_search" 
                for i in range(len(recent_actions)-1))
        )
        if browser_web_alternating:
            return f"Alternating between browser and web search without progress, trying tool creation"
    
    return ""


def _get_exploration_action(action_history: List[str], completeness: float) -> str:
    """Get the next action for exploration"""
    browser_attempts = action_history.count("browser_automation")
    web_attempts = action_history.count("web_search")
    
    if browser_attempts >= 2 and completeness < 0.5:
        return "web_search"
    elif web_attempts >= 2 and completeness < 0.5:
        return "browser_automation"
    elif len(action_history) >= 4 and completeness < 0.4:
        return "create_tools"
    
    return "synthesize"


def _build_coordinator_command(state: State, analysis: Dict[str, Any]) -> Command:
    """Build the coordinator command with enhanced analysis"""
    iteration = state.get("iteration_count", 0)
    action_history = state.get("action_history", [])
    image_files = state.get("image_files", [])
    evaluation_details = state.get("evaluation_details", {})
    
    next_action = analysis.get("next_action", "synthesize")
    goto = ACTION_MAP.get(next_action, "synthesizer")
    
    # Build streaming chunks
    chunks = _build_coordinator_chunks(
        image_files, next_action, analysis.get("reasoning", ""),
        analysis.get("search_strategy", "broader"), analysis.get("missing_info", ""),
        analysis.get("browser_capabilities_needed", []), goto, evaluation_details, action_history
    )
    
    # Store enhanced analysis
    enhanced_analysis = analysis.copy()
    enhanced_analysis.update({
        "iteration": iteration + 1,
        "considered_evaluation_feedback": bool(evaluation_details),
        "action_history": action_history
    })
    
    return Command(
        update={
            "coordinator_analysis": enhanced_analysis,
            "iteration_count": iteration + 1,
            "action_history": action_history,
            "streaming_chunks": chunks
        },
        goto=goto
    )


def _build_fallback_command(state: State, error_msg: str) -> Command:
    """Build fallback command when analysis fails"""
    iteration = state.get("iteration_count", 0)
    return Command(
        update={
            "streaming_chunks": [f"🧠 **Coordinator Error:** {error_msg}", "→ Falling back to synthesizer"],
            "iteration_count": iteration + 1
        },
        goto="synthesizer"
    )


def _build_information_analysis(web_results: List[Dict], mcp_results: List[str], browser_results: List[str]) -> str:
    """Build information quality analysis."""
    analysis_parts = []
    
    if web_results:
        high_quality_results = [r for r in web_results if r.get('credibility_score', 0.5) > 0.7]
        analysis_parts.append(f"High-quality web sources: {len(high_quality_results)}/{len(web_results)}")
        
        # Check for recent information
        recent_indicators = ['2024', '2023', 'latest', 'recent', 'today', 'current']
        recent_results = [r for r in web_results if any(indicator in r.get('content', '').lower() for indicator in recent_indicators)]
        analysis_parts.append(f"Recent information sources: {len(recent_results)}/{len(web_results)}")
    
    if mcp_results:
        analysis_parts.append(f"Computational results: {len(mcp_results)} tools executed")
    
    if browser_results:
        # Check if browser results are useful
        browser_has_useful_results = any(
            "final result:" in str(result).lower() or
            "urls visited:" in str(result).lower() or
            "actions executed:" in str(result).lower() or
            "sites:" in str(result).lower()
            for result in browser_results
        )
        status = "successful" if browser_has_useful_results else "results (may need more info)"
        analysis_parts.append(f"Browser automation: {len(browser_results)} {status}")
    
    return "\n".join(analysis_parts) if analysis_parts else "No information available"


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
        return _fallback_analysis(query)


def _fallback_analysis(query: str) -> Dict[str, Any]:
    """Fallback analysis when LLM parsing fails."""
    return {
        "next_action": "synthesize",
        "reasoning": "Fallback: LLM analysis failed, synthesizing available information",
        "confidence": 0.5
    }


def _build_coordinator_chunks(image_files: List[str], next_action: str, reasoning: str, 
                            search_strategy: str, missing_info: str, browser_capabilities: List[str], 
                            goto: str, evaluation_details: Dict[str, Any], action_history: List[str]) -> List[str]:
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
    
    # Show action history if available
    if action_history:
        recent_actions = action_history[-3:]  # Show last 3 actions
        chunks.append(f"📈 **Recent Actions:** {' → '.join(recent_actions)}")
    
    if image_files:
        chunks.append(f"🖼️ **Vision enabled:** {len(image_files)} images detected")
    
    # Check if this is a strategy switch
    is_strategy_switch = _is_strategy_switch(next_action, action_history)
    
    # Build action-specific chunks
    if next_action == "browser_automation":
        chunks.extend([
            f"🌐 **Coordinator:** {reasoning}",
            f"🤖 **Browser Capabilities:** {', '.join(browser_capabilities)}",
            "→ Routing to browser automation"
        ])
        if is_strategy_switch:
            chunks.insert(-1, "🔄 **Strategy Switch:** Multiple web search attempts failed, trying browser automation")
            
    elif next_action == "web_search":
        chunks.extend([
            f"🧠 **Coordinator:** {reasoning}",
            f"🎯 **Strategy:** {search_strategy} search",
            f"💡 **Missing:** {missing_info}",
            f"→ Routing to {goto}"
        ])
        if is_strategy_switch:
            chunks.insert(-1, "🔄 **Strategy Switch:** Multiple browser attempts failed, trying web search")
            
    else:
        chunks.extend([
            f"🧠 **Coordinator:** {reasoning}",
            f"→ Routing to {goto}"
        ])
        if is_strategy_switch and next_action == "create_tools":
            chunks.insert(-1, "🔄 **Strategy Switch:** Alternating approaches failed, trying tool creation")
    
    return chunks


def _is_strategy_switch(next_action: str, action_history: List[str]) -> bool:
    """Check if this is a strategy switch due to multiple failed attempts"""
    if not action_history:
        return False
    
    if next_action == "web_search" and action_history.count("browser_automation") >= 2:
        return True
    elif next_action == "browser_automation" and action_history.count("web_search") >= 2:
        return True
    elif next_action == "create_tools" and len(action_history) >= 4:
        return True
    
    return False 