"""
Evaluator Node Prompts

Prompts used by the evaluator node for intelligent answer completeness assessment.
"""

EVALUATOR_ANALYSIS_PROMPT = """You are evaluating whether we have enough information to provide a complete answer to the user's query.

USER QUERY: {query}

AVAILABLE INFORMATION:

Web Search Results ({web_results_count} results):
{web_results_summary}

Tool Execution Results ({mcp_results_count} results):
{mcp_results_summary}

Current iteration: {iteration}/{max_iter}

EVALUATION CRITERIA:
- Do we have enough information to answer the user's query completely?
- Is the information current and relevant?
- Are there any critical gaps in the information?
- Should we gather more information or proceed to synthesis?

Respond with a JSON object:
{{
    "completeness_score": 0.0-1.0,
    "has_sufficient_info": true/false,
    "missing_aspects": ["list", "of", "missing", "info"],
    "recommendation": "continue_search" | "synthesize",
    "reasoning": "explanation of the assessment"
}}""" 