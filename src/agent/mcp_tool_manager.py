"""
MCP Tool Manager - Utility for managing file-based MCP tools
"""

import os
import sys
import logging
import subprocess
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger('alita.mcp_tool_manager')

class MCPToolManager:
    """Manager for file-based MCP tools with utilities for viewing, editing, and managing tools"""
    
    def __init__(self, tools_dir: str = "mcp_tools"):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(exist_ok=True)
        logger.info(f"MCP Tool Manager initialized with tools directory: {self.tools_dir}")
    
    def list_tools(self, show_details: bool = False) -> List[Dict[str, Any]]:
        """List all available tool files"""
        tools = []
        
        if not self.tools_dir.exists():
            logger.warning(f"Tools directory does not exist: {self.tools_dir}")
            return tools
        
        for tool_file in self.tools_dir.glob("*.py"):
            tool_info = {
                'name': tool_file.stem,
                'file_path': str(tool_file),
                'size': tool_file.stat().st_size,
                'modified': tool_file.stat().st_mtime
            }
            
            if show_details:
                # Read file content for additional details
                try:
                    with open(tool_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract function name and description from content
                    lines = content.split('\n')
                    function_name = None
                    description = None
                    
                    for line in lines:
                        if line.strip().startswith('def ') and '(' in line:
                            function_name = line.strip().split('def ')[1].split('(')[0].strip()
                        elif line.strip().startswith('"""') or line.strip().startswith("'''"):
                            # Extract docstring
                            start = line.find('"""') if '"""' in line else line.find("'''")
                            if start != -1:
                                end = content.find('"""', start + 3) if '"""' in line else content.find("'''", start + 3)
                                if end != -1:
                                    description = content[start + 3:end].strip()
                                    break
                    
                    tool_info['function_name'] = function_name
                    tool_info['description'] = description
                    tool_info['line_count'] = len(lines)
                    
                except Exception as e:
                    logger.warning(f"Error reading tool file {tool_file}: {e}")
                    tool_info['error'] = str(e)
            
            tools.append(tool_info)
        
        return sorted(tools, key=lambda x: x['name'])
    
    def view_tool(self, tool_name: str) -> Optional[str]:
        """View the content of a specific tool"""
        tool_file = self.tools_dir / f"{tool_name}.py"
        
        if not tool_file.exists():
            logger.error(f"Tool file not found: {tool_file}")
            return None
        
        try:
            with open(tool_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return content
            
        except Exception as e:
            logger.error(f"Error reading tool file {tool_file}: {e}")
            return None
    
    def edit_tool(self, tool_name: str, editor: Optional[str] = None) -> bool:
        """Open a tool file in an editor"""
        tool_file = self.tools_dir / f"{tool_name}.py"
        
        if not tool_file.exists():
            logger.error(f"Tool file not found: {tool_file}")
            return False
        
        try:
            # Determine editor to use
            if editor:
                editor_cmd = editor
            elif 'EDITOR' in os.environ:
                editor_cmd = os.environ['EDITOR']
            elif sys.platform.startswith('win'):
                editor_cmd = 'notepad'
            else:
                # Try common editors
                for ed in ['nano', 'vim', 'vi', 'code']:
                    if subprocess.run(['which', ed], capture_output=True).returncode == 0:
                        editor_cmd = ed
                        break
                else:
                    logger.error("No suitable editor found. Please set EDITOR environment variable.")
                    return False
            
            # Open the file in the editor
            subprocess.run([editor_cmd, str(tool_file)])
            logger.info(f"Opened tool file {tool_file} in {editor_cmd}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening tool file {tool_file} in editor: {e}")
            return False
    
    def create_tool_template(self, tool_name: str, description: str = "") -> bool:
        """Create a new tool template file"""
        tool_file = self.tools_dir / f"{tool_name}.py"
        
        if tool_file.exists():
            logger.warning(f"Tool file already exists: {tool_file}")
            return False
        
        try:
            template = f"""# MCP Tool: {tool_name}
# Generated: {self._get_timestamp()}
# Source: Manual creation
# 
# This file contains the implementation of the '{tool_name}' MCP tool.
# The tool is automatically generated and managed by the Alita MCP Registry.
#

def {tool_name}(query=""):
    \"\"\"
    {description or f'Tool for: {tool_name}'}
    
    Args:
        query (str): The input query to process
        
    Returns:
        str: The processed result
    \"\"\"
    try:
        # TODO: Implement your tool logic here
        # Process the query and return the result
        
        result = f"Tool '{tool_name}' processed: {{query}}"
        return result
        
    except Exception as e:
        return f"Error in {tool_name}: {{str(e)}}"

# Example usage:
# result = {tool_name}("your input here")
# print(result)
"""
            
            with open(tool_file, 'w', encoding='utf-8') as f:
                f.write(template)
            
            logger.info(f"Created tool template: {tool_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating tool template {tool_file}: {e}")
            return False
    
    def delete_tool(self, tool_name: str) -> bool:
        """Delete a tool file"""
        tool_file = self.tools_dir / f"{tool_name}.py"
        
        if not tool_file.exists():
            logger.error(f"Tool file not found: {tool_file}")
            return False
        
        try:
            tool_file.unlink()
            logger.info(f"Deleted tool file: {tool_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting tool file {tool_file}: {e}")
            return False
    
    def backup_tools(self, backup_dir: str = "mcp_tools_backup") -> bool:
        """Create a backup of all tool files"""
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        try:
            import shutil
            
            # Copy all tool files to backup directory
            for tool_file in self.tools_dir.glob("*.py"):
                backup_file = backup_path / tool_file.name
                shutil.copy2(tool_file, backup_file)
            
            logger.info(f"Backed up {len(list(self.tools_dir.glob('*.py')))} tools to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up tools: {e}")
            return False
    
    def restore_tools(self, backup_dir: str = "mcp_tools_backup") -> bool:
        """Restore tool files from backup"""
        backup_path = Path(backup_dir)
        
        if not backup_path.exists():
            logger.error(f"Backup directory not found: {backup_path}")
            return False
        
        try:
            import shutil
            
            # Copy all backup files to tools directory
            for backup_file in backup_path.glob("*.py"):
                tool_file = self.tools_dir / backup_file.name
                shutil.copy2(backup_file, tool_file)
            
            logger.info(f"Restored {len(list(backup_path.glob('*.py')))} tools from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring tools: {e}")
            return False
    
    def validate_tool(self, tool_name: str) -> Dict[str, Any]:
        """Validate a tool file for syntax and structure"""
        tool_file = self.tools_dir / f"{tool_name}.py"
        
        if not tool_file.exists():
            return {
                'valid': False,
                'error': f"Tool file not found: {tool_file}"
            }
        
        try:
            # Read the file content
            with open(tool_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check syntax
            try:
                compile(content, str(tool_file), 'exec')
                syntax_valid = True
                syntax_error = None
            except SyntaxError as e:
                syntax_valid = False
                syntax_error = str(e)
            
            # Check for function definition
            lines = content.split('\n')
            function_found = False
            function_name = None
            
            for line in lines:
                if line.strip().startswith('def ') and '(' in line:
                    function_found = True
                    function_name = line.strip().split('def ')[1].split('(')[0].strip()
                    break
            
            # Check for docstring
            has_docstring = '"""' in content or "'''" in content
            
            # Check for error handling
            has_error_handling = 'try:' in content and 'except' in content
            
            return {
                'valid': syntax_valid and function_found,
                'syntax_valid': syntax_valid,
                'syntax_error': syntax_error,
                'function_found': function_found,
                'function_name': function_name,
                'has_docstring': has_docstring,
                'has_error_handling': has_error_handling,
                'line_count': len(lines),
                'file_size': tool_file.stat().st_size
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Error validating tool: {e}"
            }
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about all tools"""
        tools = self.list_tools(show_details=True)
        
        if not tools:
            return {
                'total_tools': 0,
                'total_size': 0,
                'average_size': 0,
                'total_lines': 0,
                'average_lines': 0
            }
        
        total_size = sum(t['size'] for t in tools)
        total_lines = sum(t.get('line_count', 0) for t in tools)
        
        return {
            'total_tools': len(tools),
            'total_size': total_size,
            'average_size': total_size / len(tools),
            'total_lines': total_lines,
            'average_lines': total_lines / len(tools),
            'tools_with_docstrings': sum(1 for t in tools if t.get('description')),
            'tools_with_error_handling': sum(1 for t in tools if self.view_tool(t['name']) and 'try:' in self.view_tool(t['name']))
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def print_tool_list(self, show_details: bool = False):
        """Print a formatted list of tools"""
        tools = self.list_tools(show_details=show_details)
        
        if not tools:
            print("No tools found.")
            return
        
        print(f"\nFound {len(tools)} tools in {self.tools_dir}:")
        print("-" * 80)
        
        for tool in tools:
            print(f"📁 {tool['name']}")
            print(f"   File: {tool['file_path']}")
            print(f"   Size: {tool['size']} bytes")
            
            if show_details:
                if 'function_name' in tool:
                    print(f"   Function: {tool['function_name']}")
                if 'description' in tool and tool['description']:
                    print(f"   Description: {tool['description'][:100]}...")
                if 'line_count' in tool:
                    print(f"   Lines: {tool['line_count']}")
            
            print()
    
    def print_tool_content(self, tool_name: str):
        """Print the content of a specific tool"""
        content = self.view_tool(tool_name)
        
        if content is None:
            print(f"❌ Tool '{tool_name}' not found or could not be read.")
            return
        
        print(f"\n📄 Content of tool '{tool_name}':")
        print("=" * 80)
        print(content)
        print("=" * 80)


def main():
    """Command-line interface for MCP Tool Manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Tool Manager - Manage file-based MCP tools")
    parser.add_argument('--tools-dir', default='mcp_tools', help='Tools directory path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all tools')
    list_parser.add_argument('--details', '-d', action='store_true', help='Show detailed information')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View tool content')
    view_parser.add_argument('tool_name', help='Name of the tool to view')
    
    # Edit command
    edit_parser = subparsers.add_parser('edit', help='Edit tool file')
    edit_parser.add_argument('tool_name', help='Name of the tool to edit')
    edit_parser.add_argument('--editor', help='Editor to use')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new tool template')
    create_parser.add_argument('tool_name', help='Name of the new tool')
    create_parser.add_argument('--description', help='Tool description')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete tool file')
    delete_parser.add_argument('tool_name', help='Name of the tool to delete')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate tool file')
    validate_parser.add_argument('tool_name', help='Name of the tool to validate')
    
    # Stats command
    subparsers.add_parser('stats', help='Show tool statistics')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup all tools')
    backup_parser.add_argument('--backup-dir', default='mcp_tools_backup', help='Backup directory')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore tools from backup')
    restore_parser.add_argument('--backup-dir', default='mcp_tools_backup', help='Backup directory')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Create manager
    manager = MCPToolManager(args.tools_dir)
    
    # Execute command
    if args.command == 'list':
        manager.print_tool_list(show_details=args.details)
    
    elif args.command == 'view':
        manager.print_tool_content(args.tool_name)
    
    elif args.command == 'edit':
        success = manager.edit_tool(args.tool_name, args.editor)
        if success:
            print(f"✅ Opened tool '{args.tool_name}' in editor")
        else:
            print(f"❌ Failed to open tool '{args.tool_name}' in editor")
    
    elif args.command == 'create':
        success = manager.create_tool_template(args.tool_name, args.description)
        if success:
            print(f"✅ Created tool template '{args.tool_name}'")
        else:
            print(f"❌ Failed to create tool template '{args.tool_name}'")
    
    elif args.command == 'delete':
        success = manager.delete_tool(args.tool_name)
        if success:
            print(f"✅ Deleted tool '{args.tool_name}'")
        else:
            print(f"❌ Failed to delete tool '{args.tool_name}'")
    
    elif args.command == 'validate':
        result = manager.validate_tool(args.tool_name)
        if result['valid']:
            print(f"✅ Tool '{args.tool_name}' is valid")
            print(f"   Function: {result['function_name']}")
            print(f"   Lines: {result['line_count']}")
            print(f"   Size: {result['file_size']} bytes")
            print(f"   Has docstring: {result['has_docstring']}")
            print(f"   Has error handling: {result['has_error_handling']}")
        else:
            print(f"❌ Tool '{args.tool_name}' is invalid")
            if 'error' in result:
                print(f"   Error: {result['error']}")
            if result.get('syntax_error'):
                print(f"   Syntax error: {result['syntax_error']}")
    
    elif args.command == 'stats':
        stats = manager.get_tool_stats()
        print(f"\n📊 Tool Statistics:")
        print(f"   Total tools: {stats['total_tools']}")
        print(f"   Total size: {stats['total_size']} bytes")
        print(f"   Average size: {stats['average_size']:.1f} bytes")
        print(f"   Total lines: {stats['total_lines']}")
        print(f"   Average lines: {stats['average_lines']:.1f}")
        print(f"   Tools with docstrings: {stats['tools_with_docstrings']}")
        print(f"   Tools with error handling: {stats['tools_with_error_handling']}")
    
    elif args.command == 'backup':
        success = manager.backup_tools(args.backup_dir)
        if success:
            print(f"✅ Backed up tools to '{args.backup_dir}'")
        else:
            print(f"❌ Failed to backup tools")
    
    elif args.command == 'restore':
        success = manager.restore_tools(args.backup_dir)
        if success:
            print(f"✅ Restored tools from '{args.backup_dir}'")
        else:
            print(f"❌ Failed to restore tools")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 