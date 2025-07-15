"""
Advanced Context Length Management

Provides utilities for managing context length across the entire agent system.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from .llm_provider import LLMProvider

logger = logging.getLogger('alita.context_manager')


class ContextManager:
    """Advanced context length management for LLM interactions."""
    
    def __init__(self, max_tokens: int = 120000):
        self.max_tokens = max_tokens
        self.llm_provider = LLMProvider()
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)."""
        if not text:
            return 0
        return len(text) // 4
    
    def check_context_length(self, content: str) -> bool:
        """Check if content exceeds token limit."""
        return self.estimate_tokens(content) > self.max_tokens
    
    def token_aware_truncation(self, text: str, max_tokens: int = 1000) -> str:
        """Truncate text while preserving important information."""
        if not text:
            return text
        
        # Rough estimation: 1 token ≈ 4 characters
        char_limit = max_tokens * 4
        
        if len(text) <= char_limit:
            return text
        
        # Split into sentences and keep most relevant ones
        sentences = text.split('. ')
        relevant_sentences = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > char_limit:
                break
            relevant_sentences.append(sentence)
            current_length += len(sentence)
        
        return '. '.join(relevant_sentences) + "..."
    
    def hierarchical_summarize(self, content: str, query: str, max_tokens: int = 1000) -> str:
        """Create hierarchical summaries to maintain context while reducing tokens."""
        if not content:
            return content
        
        if self.estimate_tokens(content) <= max_tokens:
            return content
        
        try:
            # Truncate content to reasonable size for LLM processing
            content_to_process = content[:5000] if len(content) > 5000 else content
            
            # First level: extract key points
            key_points_prompt = f"""Extract the 3-5 most important points from this content in relation to the query: "{query}"

Content: {content_to_process}

Provide only the key points, one per line. Focus on information directly relevant to the query:"""
            
            key_points_response = ""
            for chunk in self.llm_provider._make_api_call(key_points_prompt):
                key_points_response += chunk
            
            # Validate key points
            if not key_points_response or len(key_points_response.strip()) < 20:
                logger.warning("Key points extraction failed, using direct summarization")
                return self.semantic_summarize(content, query)
            
            # Second level: summarize key points
            summary_prompt = f"""Summarize these key points in relation to the query: "{query}"

Key Points:
{key_points_response}

Provide a concise summary (2-3 sentences) that directly addresses the query:"""
            
            summary_response = ""
            for chunk in self.llm_provider._make_api_call(summary_prompt):
                summary_response += chunk
            
            summary = summary_response.strip()
            
            # Validate summary quality
            if summary and len(summary) > 30 and not summary.startswith("Error"):
                return summary
            else:
                logger.warning("Hierarchical summarization produced poor result, falling back to direct summarization")
                return self.semantic_summarize(content, query)
            
        except Exception as e:
            logger.warning(f"Hierarchical summarization failed: {e}")
            return self.semantic_summarize(content, query)
    
    def semantic_summarize(self, content: str, query: str) -> str:
        """Create semantic summaries focused on query relevance."""
        if not content:
            return content
        
        # If content is already short enough, return as is
        if len(content) <= 500:
            return content
        
        try:
            # Truncate content to reasonable size for LLM processing
            content_to_summarize = content[:4000] if len(content) > 4000 else content
            
            prompt = f"""Summarize this content in relation to the query: "{query}"

Content: {content_to_summarize}

Provide a concise summary (2-3 sentences) that directly relates to the query. Focus on the most relevant information."""
            
            summary_chunks = []
            for chunk in self.llm_provider._make_api_call(prompt):
                summary_chunks.append(chunk)
            
            summary = "".join(summary_chunks).strip()
            
            # Validate summary quality
            if summary and len(summary) > 50 and not summary.startswith("Error"):
                return summary
            else:
                logger.warning("Semantic summarization produced poor result, falling back to truncation")
                return self.token_aware_truncation(content, 1000)
            
        except Exception as e:
            logger.warning(f"Semantic summarization failed: {e}")
            return self.token_aware_truncation(content, 1000)
    
    def smart_context_management(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently manage context to stay within token limits using semantic summarization."""
        
        # Calculate current token usage
        current_tokens = self.estimate_tokens(str(state))
        
        # Log context usage for monitoring
        logger.info(f"Context management: {current_tokens} tokens, limit: {self.max_tokens}")
        
        if current_tokens <= self.max_tokens:
            return state
        
        logger.warning(f"Context too large ({current_tokens} tokens), applying semantic summarization")
        
        # Get the original query for semantic summarization
        original_query = state.get("original_query", "general information")
        
        # Priority-based semantic summarization
        priorities = {
            "original_query": 1,  # Always keep
            "final_answer": 1,    # Always keep
            "coordinator_analysis": 2,
            "web_search_results": 3,
            "mcp_execution_results": 3,
            "browser_results": 3,
            "streaming_chunks": 4  # Lowest priority
        }
        
        # Apply semantic summarization based on priority
        for key, priority in sorted(priorities.items(), key=lambda x: x[1]):
            if key in state and current_tokens > self.max_tokens:
                if isinstance(state[key], list):
                    # Force summarization for all long lists, regardless of type
                    if len(state[key]) > 3:
                        if key == "web_search_results":
                            summarized = self._semantically_summarize_web_results(state[key], original_query)
                            logger.info(f"Force summarized web_search_results: {len(state[key])} -> {len(summarized)} items")
                            state[key] = summarized
                        elif key == "mcp_execution_results":
                            summarized = self._semantically_summarize_mcp_results(state[key], original_query)
                            logger.info(f"Force summarized mcp_execution_results: {len(state[key])} -> {len(summarized)} items")
                            state[key] = summarized
                        elif key == "browser_results":
                            summarized = self._semantically_summarize_browser_results(state[key], original_query)
                            logger.info(f"Force summarized browser_results: {len(state[key])} -> {len(summarized)} items")
                            state[key] = summarized
                        elif key == "streaming_chunks":
                            logger.info(f"Truncating streaming_chunks: {len(state[key])} -> 5 items")
                            state[key] = state[key][-5:]
                            for i, chunk in enumerate(state[key]):
                                if len(chunk) > 500:
                                    summarized = self.semantic_summarize(chunk, original_query)
                                    logger.info(f"Summarized streaming_chunk[{i}]: {len(chunk)} -> {len(summarized)} chars")
                                    state[key][i] = summarized
                        else:
                            # For any other long list, summarize as string list
                            summarized = self._semantically_summarize_string_list([str(item) for item in state[key]], original_query)
                            logger.info(f"Force summarized {key}: {len(state[key])} -> {len(summarized)} items")
                            state[key] = summarized
                elif isinstance(state[key], str):
                    if len(state[key]) > 1000:
                        summarized = self.semantic_summarize(state[key], original_query)
                        logger.info(f"Summarized {key}: {len(state[key])} -> {len(summarized)} chars")
                        state[key] = summarized
                elif isinstance(state[key], dict):
                    for sub_key, sub_value in state[key].items():
                        if isinstance(sub_value, str) and len(sub_value) > 500:
                            summarized = self.semantic_summarize(sub_value, original_query)
                            logger.info(f"Summarized {key}[{sub_key}]: {len(sub_value)} -> {len(summarized)} chars")
                            state[key][sub_key] = summarized
                        elif isinstance(sub_value, list) and len(sub_value) > 3:
                            summarized = self._semantically_summarize_string_list([str(item) for item in sub_value], original_query)
                            logger.info(f"Force summarized {key}[{sub_key}]: {len(sub_value)} -> {len(summarized)} items")
                            state[key][sub_key] = summarized
                
                current_tokens = self.estimate_tokens(str(state))
        
        logger.info(f"Context reduced to {current_tokens} tokens using semantic summarization")
        return state
    
    def debug_context_usage(self, state: Dict[str, Any]) -> None:
        """Debug function to monitor context usage."""
        total_tokens = self.estimate_tokens(str(state))
        logger.info(f"Total estimated tokens: {total_tokens}")
        
        for key, value in state.items():
            if isinstance(value, (str, list, dict)):
                tokens = self.estimate_tokens(str(value))
                logger.info(f"{key}: {tokens} tokens")
    
    def chunk_with_overlap(self, text: str, chunk_size: int = 8000, overlap: int = 500) -> List[str]:
        """Split text into overlapping chunks for processing."""
        if not text or len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def summarize_web_results(self, web_results: List[Dict], query: str, max_results: int = 3) -> str:
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
        if self.estimate_tokens(summary) > 2000:
            try:
                semantic_summary = self.semantic_summarize(summary, query)
                return semantic_summary
            except Exception as e:
                logger.warning(f"Semantic summarization failed: {e}")
                return self.token_aware_truncation(summary, 2000)
        
        return summary
    
    def summarize_mcp_results(self, mcp_results: List[str], max_items: int = 3, max_length_per_item: int = 200) -> str:
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
    
    def _semantically_summarize_web_results(self, web_results: List[Dict], query: str) -> List[Dict]:
        """Semantically summarize web results while preserving structure."""
        if not web_results:
            return []
        
        if len(web_results) <= 2:
            return web_results
        
        try:
            # Create a combined summary of all results
            combined_content = ""
            for i, result in enumerate(web_results):
                title = result.get('title', 'No title')
                content = result.get('content', 'No content')
                combined_content += f"Result {i+1}: {title}\n{content}\n\n"
            
            # Use hierarchical summarization
            summary = self.hierarchical_summarize(combined_content, query, 2000)
            
            # Return as a single summarized result
            return [{
                'title': f"Summarized Web Results ({len(web_results)} sources)",
                'content': summary,
                'url': 'multiple_sources'
            }]
            
        except Exception as e:
            logger.warning(f"Web results semantic summarization failed: {e}")
            return web_results[:2]  # Fallback to simple truncation
    
    def _semantically_summarize_mcp_results(self, mcp_results: List[str], query: str) -> List[str]:
        """Semantically summarize MCP execution results."""
        if not mcp_results:
            return []
        
        if len(mcp_results) <= 3:
            return mcp_results
        
        try:
            # Combine all results
            combined_results = "\n".join([f"Result {i+1}: {result}" for i, result in enumerate(mcp_results)])
            
            # Use semantic summarization
            summary = self.semantic_summarize(combined_results, query)
            
            return [f"Summarized Tool Results ({len(mcp_results)} operations): {summary}"]
            
        except Exception as e:
            logger.warning(f"MCP results semantic summarization failed: {e}")
            return mcp_results[:3]  # Fallback to simple truncation
    
    def _semantically_summarize_browser_results(self, browser_results: List[str], query: str) -> List[str]:
        """Semantically summarize browser automation results."""
        if not browser_results:
            return []
        
        if len(browser_results) <= 2:
            return browser_results
        
        try:
            # Combine all browser results
            combined_results = "\n".join([f"Browser Action {i+1}: {result}" for i, result in enumerate(browser_results)])
            
            # Use semantic summarization
            summary = self.semantic_summarize(combined_results, query)
            
            return [f"Summarized Browser Results ({len(browser_results)} actions): {summary}"]
            
        except Exception as e:
            logger.warning(f"Browser results semantic summarization failed: {e}")
            return browser_results[:2]  # Fallback to simple truncation
    
    def _semantically_summarize_string_list(self, string_list: List[str], query: str) -> List[str]:
        """Semantically summarize a list of strings."""
        if not string_list:
            return []
        
        if len(string_list) <= 3:
            return string_list
        
        try:
            # Combine all strings
            combined = "\n".join([f"Item {i+1}: {item}" for i, item in enumerate(string_list)])
            
            # Use semantic summarization
            summary = self.semantic_summarize(combined, query)
            
            return [f"Summarized List ({len(string_list)} items): {summary}"]
            
        except Exception as e:
            logger.warning(f"String list semantic summarization failed: {e}")
            return string_list[:3]  # Fallback to simple truncation


# Global context manager instance
context_manager = ContextManager() 