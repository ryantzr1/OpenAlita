import re
import sys
import logging
import traceback
from typing import Dict, Any, Tuple, Optional, Callable


# Configure logging
logger = logging.getLogger('alita.mcp_factory')

class MCPFactory:
    """Factory for creating MCP functions from LLM-generated scripts."""
    
    def __init__(self):
        # Whitelist of safe modules that can be imported
        self.safe_modules = {
            'datetime', 'time', 'json', 'math', 'random', 'uuid', 'hashlib',
            'base64', 'urllib.parse', 'urllib.request', 'requests', 'os', 're'
        }
        logger.info("MCPFactory initialized with safe modules whitelist")
    
    def create_mcp_from_script(self, function_name: str, script_content: str) -> Tuple[Optional[Callable], Dict[str, Any]]:
        """
        Create an MCP function from a script string.
        Returns (function, metadata) or (None, {}) if creation fails.
        """
        try:
            logger.info(f"Creating MCP from script for function '{function_name}'")
            
            # Parse metadata from the script
            metadata = self._parse_script_metadata(script_content)
            logger.debug(f"Parsed metadata: {metadata}")
            
            # Validate and clean the script
            cleaned_script = self._clean_script(script_content)
            if not cleaned_script:
                logger.error("Script cleaning failed - returned empty script")
                return None, {}
            
            # Create a safe execution environment
            safe_globals = self._create_safe_globals(metadata.get('requires', ''))
            
            # Add debug wrapper to check for return statements
            cleaned_script = self._add_return_check_wrapper(cleaned_script, function_name)
            print(cleaned_script)  # Debug: Print the cleaned script
            # Execute the script to define the function
            logger.debug("Executing script to define function")
            try:
                exec(cleaned_script, safe_globals)
            except SyntaxError as e:
                logger.error(f"Syntax error in script: {e}")
                logger.error(f"Line {e.lineno}: {e.text}")
                return None, {}
            except Exception as e:
                logger.error(f"Error executing script: {e}", exc_info=True)
                return None, {}
            
            # Extract the function
            actual_function_name = metadata.get('name', function_name)
            logger.debug(f"Looking for function '{actual_function_name}' in executed script")
            
            if actual_function_name in safe_globals:
                function = safe_globals[actual_function_name]
                logger.info(f"Function '{actual_function_name}' successfully created")
                return function, metadata
            else:
                all_vars = [k for k in safe_globals.keys() if not k.startswith('__')]
                logger.error(f"Function '{actual_function_name}' not found in executed script")
                logger.error(f"Available variables: {all_vars}")
                return None, {}
                
        except Exception as e:
            logger.error(f"Error creating MCP from script: {e}", exc_info=True)
            return None, {}
    
    def _add_return_check_wrapper(self, script: str, function_name: str) -> str:
        """Add wrapper to function to check for proper return values and debug execution"""
        # Find the function definition line
        lines = script.split('\n')
        func_def_line = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith(f"def {function_name}") or \
               line.strip().startswith(f"def {function_name.lower()}"):
                func_def_line = i
                break
        
        if func_def_line == -1:
            logger.warning(f"Could not find function definition for '{function_name}'")
            # Add a wrapper function at the end as a fallback
            wrapper = f"""
# Debug wrapper function
def _debug_wrapper_for_original_function(*args, **kwargs):
    import traceback
    try:
        result = {function_name}(*args, **kwargs)
        if result is None:
            print(f"WARNING: Function '{function_name}' returned None. Functions must return a value!")
            return f"Error: Function '{function_name}' did not return a value"
        return result
    except Exception as e:
        print(f"ERROR in '{function_name}': {{e}}")
        traceback.print_exc()
        return f"Error in '{function_name}': {{str(e)}}"

# Replace original function with wrapped version
{function_name} = _debug_wrapper_for_original_function
"""
            return script + "\n" + wrapper
            
        # Check if the function already has proper return statements
        has_return = False
        for line in lines[func_def_line:]:
            if line.strip().startswith("return "):
                has_return = True
                break
        
        if not has_return:
            logger.warning(f"Function '{function_name}' might lack proper return statements")
        
        # Add wrapper function
        wrapper = f"""
# Original function preserved as _original_{function_name}
_original_{function_name} = {function_name}

# Debug wrapper function
def {function_name}(*args, **kwargs):
    import traceback
    try:
        result = _original_{function_name}(*args, **kwargs)
        if result is None:
            print(f"WARNING: Function '{function_name}' returned None. Functions must return a value!")
            return f"Error: Function '{function_name}' did not return a value"
        return result
    except Exception as e:
        print(f"ERROR in '{function_name}': {{e}}")
        traceback.print_exc()
        return f"Error in '{function_name}': {{str(e)}}"
"""
        return script + "\n" + wrapper
    
    def _parse_script_metadata(self, script_content: str) -> Dict[str, Any]:
        """Parse metadata from script comments."""
        metadata = {}
        
        lines = script_content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# MCP Name:'):
                metadata['name'] = line.replace('# MCP Name:', '').strip()
            elif line.startswith('# Description:'):
                metadata['description'] = line.replace('# Description:', '').strip()
            elif line.startswith('# Arguments:'):
                metadata['args'] = line.replace('# Arguments:', '').strip()
            elif line.startswith('# Returns:'):
                metadata['returns'] = line.replace('# Returns:', '').strip()
            elif line.startswith('# Requires:'):
                metadata['requires'] = line.replace('# Requires:', '').strip()
        
        return metadata
    
    def _clean_script(self, script_content: str) -> Optional[str]:
        """Clean and validate the script content."""
        try:
            # First, let's debug what we're receiving
            logger.debug(f"Raw script content received:\n{script_content[:500]}...")
            
            # Check if we have any actual content
            if not script_content or script_content.strip() == "":
                logger.error("Received empty script content")
                return None
            
            # Remove markdown code blocks if present
            lines = script_content.split('\n')
            cleaned_lines = []
            in_code_block = False
            
            for line in lines:
                line_stripped = line.strip()
                
                # Skip markdown code block markers
                if line_stripped.startswith('```'):
                    if line_stripped.startswith('```python'):
                        in_code_block = True
                    elif line_stripped == '```':
                        in_code_block = False
                    continue
                
                # If we're not in a code block and we encounter a code block start, start tracking
                if line_stripped.startswith('```python') and not in_code_block:
                    in_code_block = True
                    continue
                
                # If we're in a code block or there are no code block markers, include the line
                if in_code_block or '```' not in script_content:
                    cleaned_lines.append(line)
            
            # Now process the cleaned lines to remove metadata comments
            code_lines = []
            metadata_lines = []
            
            for line in cleaned_lines:
                # Track metadata comments separately
                if line.strip().startswith('# MCP Name:') or \
                   line.strip().startswith('# Description:') or \
                   line.strip().startswith('# Arguments:') or \
                   line.strip().startswith('# Returns:') or \
                   line.strip().startswith('# Requires:'):
                    metadata_lines.append(line)
                    continue
                code_lines.append(line)
            
            cleaned_script = '\n'.join(code_lines).strip()
            
            # Debug: Print what we found
            logger.debug(f"Found {len(metadata_lines)} metadata lines and {len(code_lines)} code lines")
            logger.debug(f"Metadata lines: {metadata_lines}")
            logger.debug(f"Cleaned script content:\n{cleaned_script[:500]}...")
            
            # Check if the script is empty after cleaning
            if not cleaned_script or cleaned_script.isspace():
                logger.error("Script is empty after removing metadata comments and markdown")
                logger.error("This suggests the LLM only generated metadata/markdown without actual code")
                logger.error(f"Original script was:\n{script_content}")
                return None
            
            # Check if we have any non-comment, non-whitespace content
            non_empty_lines = [line.strip() for line in cleaned_script.split('\n') if line.strip() and not line.strip().startswith('#')]
            if not non_empty_lines:
                logger.error("No actual code lines found - only comments and whitespace")
                return None
            
            # Check for function definition
            has_function_def = any(line.strip().startswith('def ') for line in cleaned_script.split('\n'))
            if not has_function_def:
                logger.error("No function definition found in script")
                logger.error(f"Non-empty lines found: {non_empty_lines[:5]}...")  # Show first 5 for debugging
                return None
                
            # Check for naked return statements (return without a value)
            naked_returns = re.findall(r'\breturn\s*$', cleaned_script, re.MULTILINE)
            if naked_returns:
                logger.warning(f"Found {len(naked_returns)} naked 'return' statements without values")
                # Fix them by returning an empty string
                cleaned_script = re.sub(r'\breturn\s*$', 'return ""', cleaned_script, flags=re.MULTILINE)
                logger.info("Fixed naked return statements to return empty strings")
            
            # Basic validation - check if it's valid Python
            try:
                compile(cleaned_script, '<string>', 'exec')
                logger.debug("Script compiled successfully")
                return cleaned_script
            except SyntaxError as e:
                logger.error(f"Script has syntax error at line {e.lineno}: {e.msg}")
                logger.error(f"Problematic line: {e.text}")
                logger.error(f"Full script being compiled:\n{cleaned_script}")
                return None
                
        except Exception as e:
            logger.error(f"Error cleaning script: {e}", exc_info=True)
            return None
    
    def _create_safe_globals(self, requires: str) -> Dict[str, Any]:
        """Create a safe global environment for script execution."""
        safe_globals = {
            '__builtins__': {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
                'print': print,
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'AttributeError': AttributeError,
                '__import__': __import__,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'isinstance': isinstance,
                'type': type,
                'all': all,
                'any': any,
                'iter': iter,
                'next': next,
                'chr': chr,
                'ord': ord,
            }
        }
        
        logger.debug(f"Creating safe globals with required modules: {requires}")
        
        # Add safe modules based on requirements
        if requires:
            required_modules = [mod.strip() for mod in requires.split(',')]
            for module_name in required_modules:
                if module_name in self.safe_modules:
                    try:
                        if module_name == 'datetime':
                            import datetime
                            safe_globals['datetime'] = datetime
                        elif module_name == 'time':
                            import time
                            safe_globals['time'] = time
                        elif module_name == 'json':
                            import json
                            safe_globals['json'] = json
                        elif module_name == 'math':
                            import math
                            safe_globals['math'] = math
                        elif module_name == 'random':
                            import random
                            safe_globals['random'] = random
                        elif module_name == 'uuid':
                            import uuid
                            safe_globals['uuid'] = uuid
                        elif module_name == 'hashlib':
                            import hashlib
                            safe_globals['hashlib'] = hashlib
                        elif module_name == 'base64':
                            import base64
                            safe_globals['base64'] = base64
                        elif module_name == 'urllib.parse':
                            import urllib.parse
                            safe_globals['urllib'] = __import__('urllib')
                        elif module_name == 'urllib.request':
                            import urllib.request
                            safe_globals['urllib'] = __import__('urllib')
                        elif module_name == 'requests':
                            import requests
                            safe_globals['requests'] = requests
                        elif module_name == 'os':
                            import os
                            # Only provide safe os functions
                            safe_globals['os'] = type('SafeOS', (), {
                                'path': os.path,
                                'getcwd': os.getcwd,
                                'listdir': os.listdir,
                                'environ': os.environ,
                            })()
                        elif module_name == 're':
                            import re
                            safe_globals['re'] = re
                        
                        logger.debug(f"Added {module_name} to safe globals")
                    except ImportError:
                        logger.warning(f"Could not import required module '{module_name}'")
        
        return safe_globals