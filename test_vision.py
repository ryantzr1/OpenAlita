#!/usr/bin/env python3
"""
Test script to verify vision functionality
"""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent.llm_provider import LLMProvider

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vision():
    """Test vision functionality with the GAIA image"""
    
    # Check if image exists
    image_path = "gaia_files/9318445f-fe6a-4e1b-acbf-c68228c9906a.png"
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return False
    
    logger.info(f"Testing vision with image: {image_path}")
    
    # Create LLM provider
    llm_provider = LLMProvider()
    
    # Test prompt
    prompt = """Please analyze this image carefully and provide:

1. All fractions that use / as the fraction line that appear in the text examples (before the sample problems)
2. The ANSWERS to the sample problems (the simplified fractions, not the original problems)

The question asks for "all the fractions that use / as the fraction line and the answers to the sample problems."

Please provide a comma-separated list with no whitespace, ordered by the order in which the fractions appear in the text examples first, then the answers to the sample problems in order.

For example, if the text shows fractions like 3/4, 1/4, etc., and then there are sample problems with answers like 1/2, 1/3, etc., the format should be: 3/4,1/4,1/2,1/3"""
    
    try:
        # Test vision API call
        logger.info("Making vision API call...")
        response_chunks = []
        
        for chunk in llm_provider._make_vision_api_call(prompt, [image_path]):
            response_chunks.append(chunk)
            print(chunk, end='', flush=True)
        
        print("\n" + "="*50)
        print("FULL RESPONSE:")
        print("="*50)
        full_response = "".join(response_chunks)
        print(full_response)
        
        return True
        
    except Exception as e:
        logger.error(f"Vision test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_vision()
    if success:
        print("\n✅ Vision test completed successfully!")
    else:
        print("\n❌ Vision test failed!")
        sys.exit(1) 