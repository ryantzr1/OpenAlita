"""
Context Management Utilities

Advanced context length management utilities for the workflow nodes.
"""

import logging
from typing import Dict, Any, List
from ..context_manager import context_manager

logger = logging.getLogger('alita.context_utils')


def apply_context_management(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply smart context management to state."""
    # Monitor state size before management
    pre_tokens = context_manager.estimate_tokens(str(state))
    logger.info(f"Pre-context management: {pre_tokens} tokens")
    
    state = context_manager.smart_context_management(state)
    
    # Monitor state size after management
    post_tokens = context_manager.estimate_tokens(str(state))
    logger.info(f"Post-context management: {post_tokens} tokens")
    
    context_manager.debug_context_usage(state)
    return state


def summarize_web_results(web_results: List[Dict], query: str, llm_provider, max_results: int = 3) -> str:
    """Enhanced web results summarization with semantic relevance."""
    if not web_results:
        return "No web search results"
    
    # Limit results first
    limited_results = web_results[:max_results]
    
    # Create structured summary
    summary_parts = []
    for i, result in enumerate(limited_results, 1):
        title = result.get('title', 'No title')
        content = result.get('content', 'No content')
        
        # Truncate content if too long
        if len(content) > 300:
            content = content[:300] + "..."
        
        summary_parts.append(f"{i}. {title}\n   {content}")
    
    summary = "\n\n".join(summary_parts)
    
    # If still too long, use semantic summarization
    if context_manager.estimate_tokens(summary) > 2000:
        try:
            semantic_summary = context_manager.semantic_summarize(summary, query)
            return semantic_summary
        except Exception as e:
            logger.warning(f"Semantic summarization failed: {e}")
            return context_manager.token_aware_truncation(summary, 2000)
    
    return summary


def summarize_mcp_results(mcp_results: List[str], max_items: int = 3, max_length_per_item: int = 200) -> str:
    """Summarize and filter MCP execution results to prevent token overflow."""
    if not mcp_results:
        return "No tool execution results"
    
    # Filter and truncate results
    filtered_results = []
    total_length = 0
    max_total_length = 1000  # Overall limit
    
    for i, result in enumerate(mcp_results):
        if i >= max_items:
            break
            
        # Truncate individual results
        if len(result) > max_length_per_item:
            result = result[:max_length_per_item-3] + "..."
        
        # Check total length
        if total_length + len(result) > max_total_length:
            remaining_items = len(mcp_results) - i
            filtered_results.append(f"... and {remaining_items} more results (truncated)")
            break
            
        filtered_results.append(result)
        total_length += len(result)
    
    if not filtered_results:
        return "Tool execution completed (results too large to display)"
    
    return "\n".join(filtered_results)


def limit_browser_results(browser_results: List[str], max_results: int = 2) -> List[str]:
    """Limit browser results to prevent context overflow."""
    if not browser_results:
        return browser_results
    
    if len(browser_results) <= max_results:
        return browser_results
    
    logger.info(f"Limiting browser results from {len(browser_results)} to {max_results}")
    return browser_results[:max_results] 