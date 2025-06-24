"""
Synthesizer Node Prompts

Prompts used by the synthesizer node for final answer generation.
"""

SYNTHESIS_PROMPT = """Create a comprehensive answer for: {query}

Web Search Results:
{web_results_summary}

Tool/Browser Results:
{mcp_results_summary}

IMPORTANT INSTRUCTIONS:
- If the tool/browser results contain specific answers, data, or findings, USE THEM DIRECTLY as your answer
- Trust the tool results - they have already done the research/automation work
- Do NOT second-guess or add disclaimers about tool results
- If tools found specific information (times, names, data), present it confidently
- Only mention limitations if the tools explicitly failed or returned no data

Provide a clear, direct answer based on the available information.""" 