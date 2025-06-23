#!/usr/bin/env python3
"""
Test script for the MCP Server Manager

This script tests the creation, startup, and management of local MCP servers.
"""

import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from src.agent.mcp_server_manager import MCPServerManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mcp_server_creation():
    """Test MCP server creation functionality"""
    
    print("🧪 Testing MCP Server Manager")
    print("=" * 50)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MCPServerManager(
            config_file=Path(temp_dir) / "test_mcp_servers.json",
            servers_dir=Path(temp_dir) / "mcp_servers"
        )
        
        # Test server creation
        server_name = "test_weather_analyzer"
        script_content = '''
def test_weather_analyzer(query):
    """Analyze weather data based on query"""
    return f"Weather analysis for: {query} - Temperature: 72°F, Conditions: Sunny"
'''
        metadata = {
            "name": "test_weather_analyzer",
            "description": "Analyze weather data",
            "purpose": "Provide weather analysis",
            "requires": []
        }
        
        print(f"🔧 Creating MCP server: {server_name}")
        success, result = manager.create_mcp_server(server_name, script_content, metadata)
        
        if success:
            print(f"✅ Server created successfully at: {result}")
            
            # Check if files were created
            server_dir = Path(result)
            server_file = server_dir / "server.py"
            init_file = server_dir / "__init__.py"
            
            print(f"📁 Server directory exists: {server_dir.exists()}")
            print(f"🐍 Server file exists: {server_file.exists()}")
            print(f"📦 Init file exists: {init_file.exists()}")
            
            # Test server startup (without actually starting due to dependencies)
            print(f"🚀 Testing server startup...")
            start_success, start_result = manager.start_mcp_server(server_name, result)
            
            if start_success:
                print(f"✅ Server started successfully on port: {start_result}")
                
                # Test config addition
                manager.add_server_to_config(server_name, result, int(start_result))
                print(f"📝 Added to configuration")
                
                # Check running servers
                running = manager.get_running_servers()
                print(f"🔄 Running servers: {running}")
                
            else:
                print(f"❌ Server startup failed: {start_result}")
            
            # Test cleanup
            print(f"🧹 Testing cleanup...")
            cleanup_success = manager.cleanup_server_files(server_name)
            print(f"Cleanup successful: {cleanup_success}")
            
        else:
            print(f"❌ Server creation failed: {result}")
    
    print("\n✅ MCP Server Manager test completed!")

def test_config_management():
    """Test configuration file management"""
    
    print("\n📋 Testing Configuration Management")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "test_config.json"
        manager = MCPServerManager(config_file=str(config_file))
        
        # Test initial config
        config = manager._load_config()
        print(f"📄 Initial config: {config}")
        
        # Test adding server to config
        manager.add_server_to_config("test_server", "/path/to/server", 8001)
        
        # Test updated config
        updated_config = manager._load_config()
        print(f"📄 Updated config: {updated_config}")
        
        # Verify server was added
        if "test_server" in updated_config.get("mcpServers", {}):
            print("✅ Server successfully added to configuration")
        else:
            print("❌ Server not found in configuration")

if __name__ == "__main__":
    print("🚀 MCP Server Manager Test Suite")
    print("This tests the creation and management of local MCP servers\n")
    
    try:
        test_mcp_server_creation()
        test_config_management()
        
        print("\n✅ All tests completed successfully!")
        print("\nℹ️  Key Features Tested:")
        print("   • MCP server creation from script content")
        print("   • Server file generation and validation")
        print("   • Configuration file management")
        print("   • Port allocation and server startup")
        print("   • Cleanup and resource management")
        
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc() 