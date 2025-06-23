import re
import sys
import logging
import traceback
import subprocess
import importlib
from typing import Dict, Any, Tuple, Optional, Callable


# Configure logging
logger = logging.getLogger('alita.mcp_factory')

class MCPFactory:
    """Factory for creating MCP functions from LLM-generated scripts with automatic package installation."""
    
    def __init__(self):
        logger.info("MCPFactory initialized with automatic package installation support")
    
    def _install_package(self, package_name: str) -> bool:
        """Install a package using uv if it's not available."""
        try:
            # Validate package name - skip if it's not a valid package name
            if not package_name or package_name.strip() == "":
                logger.debug(f"Skipping empty package name")
                return False
                
            # Skip common non-package strings that LLMs might generate
            invalid_packages = [
                "no external modules required",
                "none",
                "n/a",
                "not required",
                "no requirements",
                "built-in",
                "standard library",
                "no dependencies"
            ]
            
            if package_name.lower().strip() in [p.lower() for p in invalid_packages]:
                logger.debug(f"Skipping invalid package name: '{package_name}'")
                return False
                
            # Check if it looks like a valid package name (basic validation)
            if not re.match(r'^[a-zA-Z0-9_-]+$', package_name):
                logger.debug(f"Skipping invalid package name format: '{package_name}'")
                return False
            
            logger.info(f"Attempting to install package: {package_name}")
            
            # Try uv first (faster)
            result = subprocess.run(
                ["uv", "add", package_name],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully installed {package_name} with uv")
                return True
            else:
                logger.warning(f"uv installation failed for {package_name}: {result.stderr}")
                
                # Fallback to pip
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package_name],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    logger.info(f"Successfully installed {package_name} with pip")
                    return True
                else:
                    logger.error(f"pip installation also failed for {package_name}: {result.stderr}")
                    return False
                    
        except subprocess.TimeoutExpired:
            logger.error(f"Installation timeout for package: {package_name}")
            return False
        except Exception as e:
            logger.error(f"Error installing package {package_name}: {e}")
            return False
    
    def _import_with_auto_install(self, module_name: str) -> Optional[Any]:
        """Try to import a module, installing it automatically if not found."""
        try:
            # Try importing directly first
            module = importlib.import_module(module_name)
            logger.debug(f"Successfully imported {module_name}")
            return module
        except ImportError:
            logger.info(f"Module {module_name} not found, attempting to install...")
            
            # Common package name mappings
            package_mappings = {
                'cv2': 'opencv-python',
                'PIL': 'Pillow',
                'sklearn': 'scikit-learn',
                'bs4': 'beautifulsoup4',
                'yaml': 'PyYAML',
                'requests_html': 'requests-html',
                'dateutil': 'python-dateutil'
            }
            
            # Determine package name
            package_name = package_mappings.get(module_name, module_name)
            
            # Try to install the package
            if self._install_package(package_name):
                try:
                    # Try importing again after installation  
                    module = importlib.import_module(module_name)
                    logger.info(f"Successfully imported {module_name} after installation")
                    return module
                except ImportError as e:
                    logger.error(f"Failed to import {module_name} even after installation: {e}")
                    return None
            else:
                logger.error(f"Failed to install package for module: {module_name}")
                return None

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
            logger.debug(f"Final script for {function_name}:\n{cleaned_script}")
            
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
                
                # Log function signature for debugging
                import inspect
                try:
                    sig = inspect.signature(function)
                    logger.debug(f"Function signature: {sig}")
                except Exception as sig_error:
                    logger.warning(f"Could not inspect function signature: {sig_error}")
                
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
        
        # Parse the function signature to determine if it takes parameters
        func_def_line_content = lines[func_def_line].strip()
        has_parameters = False
        
        # Check if the function definition has parameters (anything between parentheses)
        if '(' in func_def_line_content and ')' in func_def_line_content:
            param_part = func_def_line_content.split('(', 1)[1].split(')', 1)[0].strip()
            # If there are parameters (not just empty parentheses)
            if param_part and param_part != 'self':
                has_parameters = True
        
        # Check if the function already has proper return statements
        has_return = False
        for line in lines[func_def_line:]:
            if line.strip().startswith("return "):
                has_return = True
                break
        
        if not has_return:
            logger.warning(f"Function '{function_name}' might lack proper return statements")
        
        # Add wrapper function with proper parameter handling
        if has_parameters:
            # Function takes parameters - use the original wrapper
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
        else:
            # Function takes no parameters - call without arguments
            wrapper = f"""
# Original function preserved as _original_{function_name}
_original_{function_name} = {function_name}

# Debug wrapper function
def {function_name}(*args, **kwargs):
    import traceback
    try:
        # Function takes no parameters, so call without arguments
        result = _original_{function_name}()
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
            
            # Now process the cleaned lines to remove metadata comments and obvious example code
            code_lines = []
            metadata_lines = []
            
            for line in cleaned_lines:
                line_stripped = line.strip()
                
                # Track metadata comments separately
                if line_stripped.startswith('# MCP Name:') or \
                   line_stripped.startswith('# Description:') or \
                   line_stripped.startswith('# Arguments:') or \
                   line_stripped.startswith('# Returns:') or \
                   line_stripped.startswith('# Requires:'):
                    metadata_lines.append(line)
                    continue
                
                # Skip obvious example code and test cases
                if any(keyword in line_stripped.lower() for keyword in [
                    'example usage:', 'test with', 'output:', 'if __name__', 'main()', 'test_', 'example:'
                ]):
                    continue
                
                # Skip standalone print statements that are clearly examples
                if line_stripped.startswith('print(') and any(keyword in line_stripped.lower() for keyword in [
                    'output', 'result', 'example', 'test'
                ]):
                    continue
                
                # Skip result assignments that are clearly examples
                if line_stripped.startswith('result =') and any(keyword in line_stripped.lower() for keyword in [
                    'output', 'example', 'test'
                ]):
                    continue
                
                # Include everything else (be more conservative)
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
                
                # Try to fix common syntax errors
                logger.info("Attempting to fix common syntax errors...")
                fixed_script = self._fix_common_syntax_errors(cleaned_script)
                if fixed_script:
                    try:
                        compile(fixed_script, '<string>', 'exec')
                        logger.info("Script compiled successfully after fixing syntax errors")
                        return fixed_script
                    except SyntaxError as e2:
                        logger.error(f"Still has syntax error after fixing: {e2}")
                        return None
                return None
                
        except Exception as e:
            logger.error(f"Error cleaning script: {e}", exc_info=True)
            return None
    
    def _fix_common_syntax_errors(self, script: str) -> Optional[str]:
        """Fix common syntax errors in generated scripts."""
        try:
            lines = script.split('\n')
            fixed_lines = []
            i = 0
            
            while i < len(lines):
                line = lines[i]
                line_stripped = line.strip()
                
                # Skip lines that look like example code with syntax errors
                if 'print(' in line and ('"' in line or "'" in line):
                    # Count quotes to see if they're balanced
                    single_quotes = line.count("'")
                    double_quotes = line.count('"')
                    
                    # If quotes are unbalanced, skip this line (it's likely example code)
                    if single_quotes % 2 != 0 or double_quotes % 2 != 0:
                        logger.debug(f"Skipping line with unbalanced quotes: {line}")
                        i += 1
                        continue
                
                # Handle unterminated docstrings
                if '"""' in line:
                    # Count triple quotes in this line
                    triple_quotes = line.count('"""')
                    if triple_quotes % 2 != 0:
                        # Unterminated docstring - find the end
                        logger.debug(f"Found unterminated docstring in line: {line}")
                        fixed_lines.append(line)
                        i += 1
                        
                        # Look for the closing triple quotes
                        while i < len(lines):
                            next_line = lines[i]
                            fixed_lines.append(next_line)
                            if '"""' in next_line:
                                # Found closing docstring
                                break
                            i += 1
                        continue
                
                fixed_lines.append(line)
                i += 1
            
            return '\n'.join(fixed_lines)
            
        except Exception as e:
            logger.error(f"Error fixing syntax errors: {e}")
            return None
    
    def _create_safe_globals(self, requires: str) -> Dict[str, Any]:
        """Create a flexible global environment with automatic package installation."""
        # Full Python builtins - no restrictions
        import builtins
        safe_globals = {
            '__builtins__': builtins.__dict__.copy()
        }
        
        logger.debug(f"Creating globals with auto-install for required modules: {requires}")
        
        # Add required modules with automatic installation
        if requires:
            # Clean and validate the requires string
            requires_clean = requires.strip()
            
            # Skip if it's a common non-module string
            invalid_requires = [
                "no external modules required",
                "none",
                "n/a",
                "not required",
                "no requirements",
                "built-in",
                "standard library",
                "no dependencies"
            ]
            
            if requires_clean.lower() in [r.lower() for r in invalid_requires]:
                logger.debug(f"Skipping invalid requires field: '{requires_clean}'")
            else:
                # Parse comma-separated modules
                required_modules = []
                for mod in requires_clean.split(','):
                    mod_clean = mod.strip()
                    if mod_clean and mod_clean.lower() not in [r.lower() for r in invalid_requires]:
                        required_modules.append(mod_clean)
                
                logger.debug(f"Valid modules to import: {required_modules}")
                
                for module_name in required_modules:
                    logger.debug(f"Attempting to import/install module: {module_name}")
                    
                    # Handle special cases for submodules
                    if '.' in module_name:
                        # For modules like urllib.parse, requests.auth, etc.
                        base_module = module_name.split('.')[0]
                        module = self._import_with_auto_install(base_module)
                        if module:
                            safe_globals[base_module] = module
                            # Also try to import the specific submodule
                            try:
                                submodule = importlib.import_module(module_name)
                                # Add submodule with full path name
                                safe_globals[module_name.replace('.', '_')] = submodule
                            except ImportError:
                                logger.warning(f"Could not import submodule {module_name}")
                    else:
                        # Regular module import with auto-install
                        module = self._import_with_auto_install(module_name)
                        if module:
                            safe_globals[module_name] = module
                        else:
                            logger.warning(f"Failed to import/install module: {module_name}")
        
        # Always include commonly used modules
        common_modules = ['os', 're', 'sys', 'json', 'datetime', 'time', 'math', 'random', 'PIL']
        for module_name in common_modules:
            if module_name not in safe_globals:
                module = self._import_with_auto_install(module_name)
                if module:
                    safe_globals[module_name] = module
        
        logger.info(f"Created globals environment with {len(safe_globals)} available modules")
        return safe_globals