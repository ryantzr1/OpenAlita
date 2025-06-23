"""
Synthesizer Node Prompts

Prompts used by the synthesizer node for final answer generation.
"""

SYNTHESIS_PROMPT = """Create a comprehensive answer for: {query}

Web Search Results:
{web_results_summary}

Tool Results:
{mcp_results_summary}

Provide a clear, helpful answer. If images are mentioned in the context, analyze them using your vision capabilities.""" 