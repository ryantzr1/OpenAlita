#!/usr/bin/env python3
"""
Test script to verify intelligent vision task routing
"""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent.workflow.nodes import coordinator_node
from agent.workflow.state import State

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vision_routing():
    """Test that vision tasks are routed intelligently"""
    
    # Test query with image
    query = "As a comma separated list with no whitespace, using the provided image provide all the fractions that use / as the fraction line and the answers to the sample problems. Order the list by the order in which the fractions appear."
    
    # Create initial state
    state = State({
        "original_query": query,
        "iteration_count": 0,
        "max_iterations": 5,
        "web_search_results": [],
        "mcp_execution_results": [],
        "coordinator_analysis": {}
    })
    
    print("🧠 Testing coordinator routing for vision task...")
    print(f"Query: {query}")
    print("="*80)
    
    try:
        # Run coordinator
        command = coordinator_node(state)
        
        print(f"📊 **Routing Decision:** {command.goto}")
        print(f"📝 **Reasoning:** {command.update.get('coordinator_analysis', {}).get('reasoning', 'No reasoning')}")
        print(f"🎯 **Next Action:** {command.update.get('coordinator_analysis', {}).get('next_action', 'Unknown')}")
        
        if command.update.get('streaming_chunks'):
            print("\n📋 **Streaming Chunks:**")
            for chunk in command.update['streaming_chunks']:
                print(f"   {chunk}")
        
        print("\n" + "="*80)
        
        # Test different types of vision queries
        test_queries = [
            "Create a tool to analyze this image and extract mathematical formulas",
            "Search for similar fraction problems online",
            "Take a screenshot of this webpage and analyze the content",
            "Simply analyze this image and tell me what fractions you see"
        ]
        
        for i, test_query in enumerate(test_queries, 1):
            print(f"\n🧪 Test {i}: {test_query}")
            print("-" * 60)
            
            test_state = State({
                "original_query": test_query,
                "iteration_count": 0,
                "max_iterations": 5,
                "web_search_results": [],
                "mcp_execution_results": [],
                "coordinator_analysis": {}
            })
            
            test_command = coordinator_node(test_state)
            print(f"   → Routed to: {test_command.goto}")
            print(f"   → Reasoning: {test_command.update.get('coordinator_analysis', {}).get('reasoning', 'No reasoning')[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Vision routing test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_vision_routing()
    if success:
        print("\n✅ Vision routing test completed successfully!")
    else:
        print("\n❌ Vision routing test failed!")
        sys.exit(1) 