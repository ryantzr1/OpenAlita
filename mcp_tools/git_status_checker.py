# MCP Tool: git_status_checker
# Generated: 2025-07-04T01:21:01.085026
# Source: File-based tool generation
# 
# This file contains the implementation of the 'git_status_checker' MCP tool.
# The tool is automatically generated and managed by the Alita MCP Registry.
#

# MCP Name: git_status_checker
# Description: Executes 'git status' command and returns the output
# Arguments: query (string) - the user query to process
# Returns: string containing git status output
# Requires: subprocess

import subprocess

def git_status_checker(query=""):
    try:
        # Execute git status command and capture output
        result = subprocess.run(['git', 'status'], 
                              capture_output=True, 
                              text=True)
        
        # If command was successful, return the output
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Git status error: {result.stderr.strip()}"
            
    except FileNotFoundError:
        return "Error: Git is not installed or not in PATH"
    except Exception as e:
        return f"Error: {str(e)}"
# Example usage