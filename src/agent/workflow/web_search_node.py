"""
Web Agent Node

Handles web search and information gathering.
"""

import json
import logging
import os
from typing import Dict, Any, Literal, List
from langgraph.types import Command

from ..llm_provider import LLMProvider
from ..web_agent import WebSearchAgent
from ..prompts import TARGETED_SEARCH_PROMPT, VERIFICATION_SEARCH_PROMPT, BROADER_SEARCH_PROMPT
from .context_utils import apply_context_management
from .state import State

logger = logging.getLogger('alita.web_agent')


def web_agent_node(state: State) -> Command[Literal["evaluator"]]:
    """Web agent node with intelligent query decomposition and strategic searching"""
    logger.info("Web agent analyzing and searching...")
    
    # Apply smart context management first
    state = apply_context_management(state)
    
    web_agent = WebSearchAgent()
    llm_provider = LLMProvider()
    query = state["original_query"]
    coordinator_analysis = state.get("coordinator_analysis", {})
    search_strategy = coordinator_analysis.get("search_strategy", "broader")
    missing_info = coordinator_analysis.get("missing_info", "")
    existing_results = state.get("web_search_results", [])
    
    # Get image files from state instead of detecting them independently
    image_files = state.get("image_files", [])
    
    # Check if this is a fallback from browser agent
    browser_results = state.get("browser_results", [])
    is_browser_fallback = any("browser automation failed" in str(result) for result in browser_results)
    
    try:
        # Create strategy-aware search prompt
        existing_results_summary = json.dumps([{'title': r.get('title', ''), 'type': r.get('search_type', 'original')} for r in existing_results[:3]], indent=2) if existing_results else "No previous results"
        
        decomposition_prompt = _build_search_prompt(query, search_strategy, missing_info, existing_results_summary, image_files)
        
        # Get search query breakdown with vision support
        decomp_response = ""
        for chunk in llm_provider._make_api_call(decomposition_prompt, image_files):
            decomp_response += chunk
        
        # Parse the JSON response
        search_queries = _parse_search_queries(decomp_response)
        
        # Adapt search approach based on strategy
        chunks = _build_web_agent_chunks(image_files, search_strategy, missing_info, search_queries)
        
        # Perform multiple searches
        all_results = _perform_searches(web_agent, search_queries, search_strategy, query, chunks)
        
        # Remove duplicates by URL
        unique_results = _remove_duplicates(all_results)
        
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


def _build_search_prompt(query: str, search_strategy: str, missing_info: str, 
                        existing_results_summary: str, image_files: List[str]) -> str:
    """Build the appropriate search prompt based on strategy."""
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

    return decomposition_prompt


def _parse_search_queries(decomp_response: str) -> List[str]:
    """Parse search queries from LLM response."""
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
    
    return search_queries


def _build_web_agent_chunks(image_files: List[str], search_strategy: str, 
                           missing_info: str, search_queries: List[str]) -> List[str]:
    """Build streaming chunks for web agent output."""
    chunks = []
    if image_files:
        chunks.append(f"🖼️ **Vision-enabled search:** {len(image_files)} images detected")
    
    if search_strategy == "targeted":
        final_search_queries = search_queries[:3]
        chunks.extend([
            f"🌐 **Web Agent:** Performing {len(final_search_queries)} TARGETED searches",
            f"🎯 **Focus:** {missing_info}"
        ])
    elif search_strategy == "verification":
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
    
    return chunks


def _perform_searches(web_agent: WebSearchAgent, search_queries: List[str], 
                     search_strategy: str, query: str, chunks: List[str]) -> List[Dict]:
    """Perform the actual web searches."""
    all_results = []
    
    if search_strategy == "targeted":
        final_search_queries = search_queries[:3]
    elif search_strategy == "verification":
        final_search_queries = search_queries[:3]
    else:
        # Broader strategy - include original query
        search_queries = search_queries[:3]  # Limit to 3 focused queries
        final_search_queries = [query] + search_queries
    
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
    
    return all_results


def _remove_duplicates(all_results: List[Dict]) -> List[Dict]:
    """Remove duplicate results by URL."""
    seen_urls = set()
    unique_results = []
    for result in all_results:
        url = result.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    return unique_results 