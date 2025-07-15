"""
Evaluator Node Prompts

Prompts used by the evaluator node for intelligent answer completeness assessment.
"""

COMPREHENSIVE_EVALUATION_PROMPT = """Comprehensively evaluate if we have enough information to answer the user's query.

USER QUERY: {query}

PREVIOUS ACTION: {previous_action}

AVAILABLE INFORMATION:

Web Search Results ({web_results_count} results):
{web_results_summary}

Tool Execution Results ({mcp_results_count} results):
{mcp_results_summary}

Browser Automation Results ({browser_results_count} results):
{browser_results_summary}

Browser Analysis:
- Has useful results: {browser_has_useful_results}
- Has failures: {browser_has_failures}
- Has final results: {browser_has_final_results}
- Has actions: {browser_has_actions}

Current iteration: {iteration}/{max_iter}

EVALUATION CRITERIA:
1. **Relevance**: Do the results directly address the user's query?
2. **Completeness**: Do we have all the information needed to answer completely?
3. **Quality**: Are the results reliable, accurate, and current?
4. **Integration**: How well do different result types work together?
5. **Progress**: Has the previous action improved our understanding?
6. **Gaps**: What specific information is still missing?

BROWSER AUTOMATION ASSESSMENT:
- If browser automation provided a FINAL RESULT, assess whether it directly answers the query
- Consider the quality and completeness of browser-obtained information
- Evaluate if browser automation was the right approach for this query
- If browser automation failed, determine if a different approach would work better

PREVIOUS ACTION ANALYSIS:
- Consider what the previous action was trying to accomplish
- Assess whether that action was successful or needs to be retried
- Determine if a different approach would be more effective

Respond with JSON:
{{
    "completeness_score": 0.0-1.0,
    "has_sufficient_info": true/false,
    "reasoning": "detailed explanation of the assessment including browser automation analysis",
    "missing_aspects": ["list", "of", "missing", "information"],
    "recommended_action": "synthesize" | "web_search" | "browser_automation" | "create_tools",
    "quality_assessment": {{
        "web_results_quality": 0.0-1.0,
        "mcp_results_quality": 0.0-1.0,
        "browser_results_quality": 0.0-1.0,
        "overall_integration": 0.0-1.0
    }},
    "previous_action_success": true/false,
    "browser_automation_assessment": {{
        "was_appropriate": true/false,
        "result_quality": 0.0-1.0,
        "should_retry": true/false,
        "alternative_approach": "web_search" | "create_tools" | "synthesize"
    }},
    "next_steps": "specific recommendations for what to do next"
}}""" 