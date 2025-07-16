"""
LangGraph Workflow State Definition

Defines the state structure for the LangGraph workflow.
"""

from typing import Dict, Any, List, Annotated
from langgraph.graph import MessagesState
from operator import add


class State(MessagesState):
    """State for the LangGraph workflow"""
    original_query: str = ""
    iteration_count: int = 0
    max_iterations: int = 5
    
    # Image files - shared across all nodes
    image_files: List[str] = []
    
    # Agent outputs
    coordinator_analysis: Dict[str, Any] = {}
    web_search_results: List[Dict[str, Any]] = []
    mcp_tools_created: List[Dict[str, Any]] = []
    mcp_execution_results: List[str] = []
    browser_results: List[str] = []
    
    # Action tracking
    action_history: List[str] = []
    
    # Evaluation and synthesis
    answer_completeness: float = 0.0
    final_answer: str = ""
    
    # Streaming - properly annotated for multiple values
    streaming_chunks: Annotated[List[str], add] = [] 