"""
Synthesizer Node

Handles final answer generation.
"""

import logging
from typing import Dict, Any, List
from langgraph.types import Command
from ..llm_provider import LLMProvider
from ..prompts import SYNTHESIS_PROMPT
from .context_utils import apply_context_management, summarize_web_results, summarize_mcp_results
from .state import State
import os

logger = logging.getLogger('alita.synthesizer')

def synthesizer_node(state: State):
    """Synthesizer node for final answer generation with natural vision support"""
    logger.info("Synthesizer creating final answer...")
    
    # Apply smart context management first
    state = apply_context_management(state)
    
    llm_provider = LLMProvider()
    query = state["original_query"]
    web_results = state.get("web_search_results", [])
    mcp_results = state.get("mcp_execution_results", [])
    browser_results = state.get("browser_results", [])
    
    # Get image files from state instead of detecting them independently
    image_files = state.get("image_files", [])
    
    # Create synthesis prompt with enhanced summarization
    web_results_summary = summarize_web_results(web_results[:3], query, llm_provider) if web_results else "No web results"
    mcp_results_summary = summarize_mcp_results(mcp_results) if mcp_results else "No tool results"
    browser_results_summary = summarize_mcp_results(browser_results) if browser_results else "No browser results"
    
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

Browser Results:
{browser_results_summary}

Please analyze the image and provide your answer in the exact format requested by the question."""
    else:
        # Use the standard prompt for non-image tasks
        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            web_results_summary=web_results_summary,
            mcp_results_summary=mcp_results_summary,
            browser_results_summary=browser_results_summary
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
        
        logger.info(f"Synthesizer created final answer: {final_answer[:100]}...")
        
        return Command(
            update={
                "final_answer": final_answer,
                "streaming_chunks": chunks
            }
        )
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        
        return Command(
            update={
                "final_answer": f"Error generating response: {str(e)}",
                "streaming_chunks": [f"❌ **Synthesis error:** {str(e)}\n"]
            }
        ) 