import shlex
import re
import inspect
import logging
import traceback
from typing import Dict, Any, List, Generator, Union, Optional, Tuple

from .llm_provider import LLMProvider
from .mcp_factory import MCPFactory
from .web_agent import WebAgent
from .langgraph_workflow import LangGraphCoordinator

# Configure logging
logger = logging.getLogger('alita.agent')

MULTIMEDIA_RESPONSE_PROMPT = "The tool '{tool_name}' has returned a multimedia response."

class AlitaAgent:
    def __init__(self):
        logger.info("Initializing AlitaAgent components")
        self.llm_provider = LLMProvider()
        self.mcp_factory = MCPFactory()
        self.web_agent = WebAgent()
        
        # Initialize the LangGraph coordinator
        self.langgraph_coordinator = LangGraphCoordinator()
        
        logger.info("AlitaAgent initialized successfully with LangGraph coordinator")

    def process_command_streaming(self, command_str: str) -> Generator[str, None, None]:
        """Process a command with streaming output using LangGraph workflow."""
        if not command_str.strip():
            yield "Please enter a command."
            return

        logger.info(f"Processing command with LangGraph workflow: '{command_str}'")

        # Handle special commands first
        if command_str.lower().strip() == "help":
            help_text = """Available commands:
- Type any natural language command and I'll intelligently route it through multiple agents
- 'quit' - Exit the application
- 'log' - View recent log entries

Examples:
- add 10 20
- greet John  
- weather in Paris
- how many stars does microsoft/vscode have
- what's the latest news about AI
- calculate circle area with radius 5

The system will automatically decide whether to use web search, create custom tools, or combine multiple approaches to best answer your query."""
            logger.info("Help command executed")
            yield help_text
            return
        
        if command_str.lower().strip() == "quit":
            logger.info("Quit command received")
            yield "quit_signal"
            return

        # Handle basic arithmetic commands directly (built-in functionality)
        if self._is_basic_arithmetic(command_str):
            result = self._handle_basic_arithmetic(command_str)
            if result:
                yield result
                return

        # Use LangGraph coordinator for intelligent processing
        logger.info("Delegating to LangGraph coordinator for intelligent workflow processing")
        yield "🚀 Starting intelligent analysis workflow...\n\n"
        
        try:
            # Stream the LangGraph workflow execution
            for chunk in self.langgraph_coordinator.process_query_streaming(command_str):
                yield chunk
                
        except Exception as e:
            error_msg = f"Error in workflow processing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield f"\n{error_msg}\n\nFalling back to simple MCP creation..."
            
            # Fallback to original simple MCP creation if LangGraph fails
            try:
                yield from self._fallback_mcp_creation_streaming(command_str)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                yield f"\nI apologize, but I encountered difficulties processing your request: {str(fallback_error)}"

    def _fallback_mcp_creation_streaming(self, command_str: str) -> Generator[str, None, None]:
        """Fallback to original MCP creation if LangGraph fails."""
        logger.info("Using fallback MCP creation")
        
        # Generate MCP name
        tokens = self._parse_command_tokens(command_str)
        potential_mcp_name = self._generate_smart_mcp_name(command_str, tokens)
        
        yield f"🔧 Creating fallback tool: '{potential_mcp_name}'...\n\n"
        
        for stream_item in self._attempt_dynamic_mcp_creation_streaming(command_str, potential_mcp_name):
            # Check if this is the special signal that MCP was created
            if isinstance(stream_item, tuple) and stream_item[0] == "__MCP_CREATED__":
                mcp_data = stream_item[1]
                mcp_display_name = stream_item[2]
                
                # Execute the newly created MCP
                try:
                    yield "\n" + "─" * 30 + "\n"
                    
                    # Set the current command context for the parameterized function
                    import builtins
                    original_builtin_command = getattr(builtins, '_current_user_command', None)
                    builtins._current_user_command = command_str
                    
                    try:
                        output = mcp_data["function"]()
                    finally:
                        # Restore the original command context
                        if original_builtin_command is not None:
                            builtins._current_user_command = original_builtin_command
                        else:
                            if hasattr(builtins, '_current_user_command'):
                                delattr(builtins, '_current_user_command')
                    
                    if output is None:
                        yield f"Error: Tool '{mcp_display_name}' returned None."
                        return
                        
                    yield output
                    return
                    
                except Exception as e:
                    yield f"Error executing tool '{mcp_display_name}': {e}"
                    return
            else:
                # Stream the LLM generation process
                yield stream_item
        
        yield "Failed to create fallback tool."

    def _is_basic_arithmetic(self, command_str: str) -> bool:
        """Check if this is a basic arithmetic command we can handle directly."""
        # Simple pattern matching for basic arithmetic
        import re
        arithmetic_pattern = r'^(add|subtract|multiply|divide)\s+\d+(\.\d+)?\s+\d+(\.\d+)?$'
        return bool(re.match(arithmetic_pattern, command_str.strip(), re.IGNORECASE))

    def _handle_basic_arithmetic(self, command_str: str) -> str:
        """Handle basic arithmetic operations directly."""
        try:
            parts = command_str.strip().split()
            if len(parts) != 3:
                return None
            
            operation = parts[0].lower()
            a = float(parts[1])
            b = float(parts[2])
            
            if operation == 'add':
                result = a + b
                return f"The sum of {a} and {b} is {result}"
            elif operation == 'subtract':
                result = a - b
                return f"The difference of {a} and {b} is {result}"
            elif operation == 'multiply':
                result = a * b
                return f"The product of {a} and {b} is {result}"
            elif operation == 'divide':
                if b == 0:
                    return "Error: Cannot divide by zero"
                result = a / b
                return f"The quotient of {a} and {b} is {result}"
            
            return None
        except (ValueError, IndexError):
            return None

    def _attempt_dynamic_mcp_creation_streaming(self, command_str: str, mcp_name: str = None) -> Generator[Union[str, Tuple[str, Dict, str]], None, None]:
        """Attempt to dynamically create an MCP for the given command."""
        try:
            # First, parse the command into tokens
            logger.info(f"Parsing command: '{command_str}'")
            tokens = self._parse_command_tokens(command_str)
            if not tokens:
                logger.warning("No valid tokens found in command")
                yield "No valid command tokens found. Please try again."
                return
            
            # Use provided MCP name or generate one
            if mcp_name is None:
                yield "🤖 Analyzing command to generate an appropriate tool name...\n"
                mcp_name = self._generate_smart_mcp_name(command_str, tokens)
                logger.info(f"Generated MCP name '{mcp_name}' for command: '{command_str}'")
            else:
                logger.info(f"Using provided MCP name '{mcp_name}' for command: '{command_str}'")
            
            # Always generate a new MCP script using LLM
            logger.info("Generating MCP script using LLM")
            yield f"🧠 Creating a fresh '{mcp_name}' tool...\n\n"
            
            for progress_chunk in self.llm_provider.generate_mcp_script_streaming(mcp_name, command_str):
                yield progress_chunk
            
            final_script = self.llm_provider.get_last_generated_mcp_script()
            
            if not final_script:
                logger.error("Failed to generate MCP script: Empty result returned")
                yield "Failed to generate a tool script. Please try again with a more specific command."
                return
            
            # Create MCP function from the generated script
            logger.info("Creating MCP function from generated script")
            mcp_function, mcp_metadata, cleaned_script = self.mcp_factory.create_mcp_from_script(mcp_name, final_script)
            
            if not mcp_function:
                logger.error("Failed to create MCP function from script")
                yield "Failed to create a working tool. The generated code had issues. Please try again."
                return
            
            # Debug - Log the actual function code
            logger.debug(f"Created MCP function: {mcp_function.__code__}")
            
            # Get display name from metadata or use the generated name
            mcp_display_name = mcp_metadata.get('name', mcp_name)
            logger.info(f"Using display name '{mcp_display_name}' for new MCP")
            
            # Create MCP data structure (but don't store it anywhere)
            mcp_data = {
                "function": mcp_function,
                "description": mcp_metadata.get('description', 'No description'),
                "args_info": mcp_metadata.get('args', 'N/A'),
                "returns_info": mcp_metadata.get('returns', 'N/A'),
                "source": "dynamically-generated",
                "requires": mcp_metadata.get('requires'),
                "script_content": cleaned_script or final_script,
                "original_command": command_str
            }
            
            logger.info(f"Successfully created fresh MCP '{mcp_display_name}' (not stored)")
            yield (f"__MCP_CREATED__", mcp_data, mcp_display_name)
        
        except Exception as e:
            error_msg = f"Error creating tool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            logger.error(f"Stack trace: {traceback.format_exc()}")
            yield error_msg

    def _parse_command_tokens(self, command_str: str) -> List[str]:
        """Parse the command string into tokens."""
        try:
            tokens = shlex.split(command_str)
            logger.debug(f"Parsed command '{command_str}' into tokens: {tokens}")
            return tokens
        except ValueError:
            # Handle quoted strings that aren't closed
            logger.warning(f"tokens for '{command_str}', falling back to simple split")
            return command_str.split()

    def _generate_smart_mcp_name(self, command_str: str, tokens: List[str]) -> str:
        """Use LLM to generate a smart, reusable MCP name based on command intent."""
        try:
            # Use LLM to analyze the command and generate an appropriate MCP name
            prompt = f"""Analyze this user command and generate a generic, reusable MCP (Module Context Protocol) name.

User Command: "{command_str}"

Rules for MCP naming:
1. The name should be GENERIC and REUSABLE for similar commands with different parameters
2. Use snake_case format (lowercase with underscores)
3. Focus on the ACTION/INTENT, not specific parameters
4. Keep it concise (2-4 words max)
5. Make it descriptive of what the tool does

Examples:
- "how many stars does ryantzr1/OpenAlita have?" → "github_stars"
- "weather in Paris" → "weather_lookup" 
- "greet John" → "greet_person"
- "calculate circle area with radius 5" → "calculate_circle_area"
- "convert 100 USD to EUR" → "currency_converter"
- "what time is it in Tokyo" → "world_time"
- "find restaurants near me" → "find_restaurants"

Generate ONLY the MCP name (no explanation, no quotes, just the name):"""

            # Get response from LLM
            response_chunks = []
            for chunk in self.llm_provider._make_api_call(prompt):
                if isinstance(chunk, str) and chunk.startswith("Error:"):
                    logger.warning(f"LLM error generating MCP name: {chunk}")
                    break
                response_chunks.append(chunk)
            
            if response_chunks:
                mcp_name = "".join(response_chunks).strip()
                # Clean up the response - remove quotes, extra text, etc.
                mcp_name = re.sub(r'[^a-zA-Z0-9_]', '', mcp_name.split('\n')[0])
                mcp_name = mcp_name.lower()
                
                # Validate the name
                if mcp_name and len(mcp_name) > 0 and len(mcp_name) < 50:
                    logger.info(f"LLM generated MCP name: '{mcp_name}' for command: '{command_str}'")
                    return mcp_name
                else:
                    logger.warning(f"LLM generated invalid MCP name: '{mcp_name}', using fallback")
            
        except Exception as e:
            logger.warning(f"Error using LLM to generate MCP name: {e}")
        
        # Fallback: simple extraction if LLM fails
        return self._fallback_mcp_name(command_str, tokens)
    
    def _fallback_mcp_name(self, command_str: str, tokens: List[str]) -> str:
        """Fallback method for generating MCP names when LLM fails."""
        # Simple fallback - take first meaningful word + action type
        meaningful_words = [word.lower() for word in tokens[:3] if len(word) > 2 and word.lower() not in ['the', 'a', 'an', 'is', 'are', 'do', 'does']]
        if meaningful_words:
            return '_'.join(meaningful_words[:2]).replace('-', '_').replace('.', '_')
        else:
            return 'custom_tool'