#!/usr/bin/env python3
"""
MCP Tools CLI - Simple command-line interface for managing MCP tools
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent.mcp_tool_manager import main

if __name__ == "__main__":
    main() 