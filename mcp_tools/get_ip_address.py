# MCP Tool: get_ip_address
# Generated: 2025-07-04T01:18:47.875972
# Source: File-based tool generation
# 
# This file contains the implementation of the 'get_ip_address' MCP tool.
# The tool is automatically generated and managed by the Alita MCP Registry.
#

# MCP Name: get_ip_address
# Description: Retrieves the current machine's IP address and returns it as a string
# Arguments: query (string) - the user query to process
# Returns: string containing the IP address
# Requires: socket

import socket

def get_ip_address(query=""):
    try:
        # Create a socket connection to determine the IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Connect to an external server (doesn't actually send any data)
        s.connect(("8.8.8.8", 80))
        
        # Get the local IP address from the socket
        ip_address = s.getsockname()[0]
        
        # Close the socket
        s.close()
        
        return ip_address
    except Exception as e:
        return f"Error: {str(e)}"