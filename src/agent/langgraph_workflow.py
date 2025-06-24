"""
LangGraph Workflow Coordinator for Open-Alita Agent

Clean implementation following best practices with proper state handling.
"""

import logging
import time
from typing import Generator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .workflow import (
    State,
    coordinator_node,
    web_agent_node,
    mcp_agent_node,
    browser_agent_node,
    browser_agent_router,
    evaluator_node,
    synthesizer_node
)

logger = logging.getLogger('alita.langgraph')


def _build_workflow():
    """Build the LangGraph workflow"""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("web_agent", web_agent_node) 
    workflow.add_node("mcp_agent", mcp_agent_node)
    workflow.add_node("browser_agent", browser_agent_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    # Add edges
    workflow.add_edge(START, "coordinator")
    workflow.add_edge("web_agent", "evaluator")
    workflow.add_edge("mcp_agent", "evaluator") 
    workflow.add_conditional_edges("browser_agent", browser_agent_router, {
        "evaluator": "evaluator",
        "web_agent": "web_agent"
    })
    workflow.add_conditional_edges("evaluator", lambda state: "synthesizer" if state.get("answer_completeness", 0) >= 0.7 else "coordinator")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile(checkpointer=MemorySaver())


class LangGraphCoordinator:
    """Main coordinator class using the clean LangGraph pattern"""
    
    def __init__(self):
        self.workflow = _build_workflow()
        logger.info("LangGraph Coordinator initialized")
    
    def process_query_streaming(self, query: str, specific_image_file: str = None) -> Generator[str, None, None]:
        """Process a query through the LangGraph workflow with streaming output"""
        import os  # Ensure os is always available
        
        # Handle image files - only include when specifically relevant
        image_files = []
        if specific_image_file:
            # Use the specific image file provided (e.g., from GAIA question)
            if os.path.exists(specific_image_file):
                image_files = [specific_image_file]
                logger.info(f"Using specific image file: {specific_image_file}")
            else:
                logger.warning(f"Specific image file not found: {specific_image_file}")
        else:
            # For non-GAIA queries, only include images if the query explicitly mentions them
            # or if there's a clear visual component to the question
            query_lower = query.lower()
            visual_keywords = ['image', 'picture', 'photo', 'screenshot', 'visual', 'chart', 'graph', 'diagram', 'map']
            has_visual_component = any(keyword in query_lower for keyword in visual_keywords)
            
            if has_visual_component:
                # Only then check for image files in the directory
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                gaia_files_dir = os.path.join(project_root, "gaia_files")
                if os.path.exists(gaia_files_dir):
                    available_images = [os.path.join(gaia_files_dir, f) for f in os.listdir(gaia_files_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
                    if available_images:
                        # For now, use the first available image if query mentions visual content
                        # In a more sophisticated implementation, you could use LLM to determine which image is most relevant
                        image_files = [available_images[0]]
                        logger.info(f"Query mentions visual content, using image: {os.path.basename(available_images[0])}")
            else:
                logger.info("Query does not mention visual content, proceeding without images")
        
        # Initialize state
        initial_state = {
            "original_query": query,
            "messages": [],
            "streaming_chunks": [],
            "image_files": image_files  # Include image files in initial state (empty if not relevant)
        }
        
        # Create unique thread ID
        thread_id = f"thread_{hash(query)}_{int(time.time())}"
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Stream the workflow execution
            for step_output in self.workflow.stream(initial_state, config=config):
                for node_name, state_update in step_output.items():
                    # Stream any chunks from this step
                    if isinstance(state_update, dict) and "streaming_chunks" in state_update:
                        for chunk in state_update["streaming_chunks"]:
                            yield chunk
                    
                    # Also yield step completion
                    yield f"✅ {node_name} completed\n"
            
            # Get final state and yield final answer
            final_state = self.workflow.get_state(config).values
            if final_state.get("final_answer"):
                yield f"\n📋 **Final Answer:**\n{final_state['final_answer']}\n"
                yield f"\n🎯 **Confidence:** {final_state.get('confidence_score', 0):.1%}\n"
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            yield f"\n❌ **Workflow error:** {str(e)}\n" 