"""
Prompts Package

Centralized location for all LangGraph workflow prompts.
"""

from .coordinator_prompts import COORDINATOR_ANALYSIS_PROMPT
from .web_agent_prompts import (
    TARGETED_SEARCH_PROMPT,
    VERIFICATION_SEARCH_PROMPT,
    BROADER_SEARCH_PROMPT
)
from .mcp_agent_prompts import (
    TOOL_REQUIREMENTS_ANALYSIS_PROMPT,
    TOOL_SCRIPT_GENERATION_PROMPT,
    BROWSER_MCP_ANALYSIS_PROMPT,
    BROWSER_TOOL_SCRIPT_PROMPT
)
from .evaluator_prompts import COMPREHENSIVE_EVALUATION_PROMPT
from .synthesizer_prompts import SYNTHESIS_PROMPT

__all__ = [
    # Coordinator prompts
    'COORDINATOR_ANALYSIS_PROMPT',
    
    # Web agent prompts
    'TARGETED_SEARCH_PROMPT',
    'VERIFICATION_SEARCH_PROMPT',
    'BROADER_SEARCH_PROMPT',
    
    # MCP agent prompts
    'TOOL_REQUIREMENTS_ANALYSIS_PROMPT',
    'TOOL_SCRIPT_GENERATION_PROMPT',
    'BROWSER_MCP_ANALYSIS_PROMPT',
    'BROWSER_TOOL_SCRIPT_PROMPT',
    
    # Evaluator prompts
    'COMPREHENSIVE_EVALUATION_PROMPT',
    
    # Synthesizer prompts
    'SYNTHESIS_PROMPT',
] 