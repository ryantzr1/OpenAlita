import shlex
import re
import inspect
import logging
import traceback
import asyncio
from typing import Dict, Any, List, Generator, Union, Optional, Tuple

from .llm_provider import LLMProvider
from .mcp_box import MCPBox
from .mcp_factory import MCPFactory
from .web_agent import WebAgent
from .langgraph_workflow import LangGraphWorkflowManager

# Configure logging
logger = logging.getLogger('alita.agent')

MULTIMEDIA_RESPONSE_PROMPT = "The tool '{tool_name}' has returned a multimedia response."

class AlitaAgent:
    def __init__(self):
        logger.info("Initializing AlitaAgent components")
        self.mcp_box = MCPBox()
        self.llm_provider = LLMProvider()
        self.mcp_factory = MCPFactory()
        self.web_agent = WebAgent()
        
        # Initialize LangGraph workflow manager
        logger.info("Initializing LangGraph workflow system")
        self.langgraph_manager = LangGraphWorkflowManager(
            self.llm_provider, 
            self.mcp_box, 
            self.web_agent, 
            self.mcp_factory
        )
        
        logger.info("Preloading basic MCPs")
        self._preload_basic_mcps()
        logger.info("AlitaAgent initialized successfully with LangGraph support")

    def _preload_basic_mcps(self):
        """Preload some basic MCPs."""
        try:
            # Example of a basic MCP: simple addition
            def add(a, b):
                return f"The sum of {a} and {b} is {float(a) + float(b)}"
            
            self.mcp_box.register_mcp(
                name="add",
                function=add,
                metadata={
                    "description": "Add two numbers",
                    "args": "a: first number, b: second number",
                    "returns": "The sum as a string",
                    "example": "add 5 3"
                }
            )
            logger.info("Basic MCPs preloaded successfully")
        except Exception as e:
            logger.error(f"Error preloading basic MCPs: {e}", exc_info=True)
            raise

    def process_command_streaming(self, command_str: str) -> Generator[str, None, None]:
        """Process a command with streaming output."""
        if not command_str.strip():
            yield "Please enter a command."
            return

        logger.info(f"Processing command: '{command_str}'")

        if command_str.lower().strip() == "help":
            mcp_list = self.mcp_box.list_mcps()
            logger.info("Help command executed")
            yield mcp_list
            return
        
        if command_str.lower().strip() == "quit":
            logger.info("Quit command received")
            yield "quit_signal"
            return

        # First, try to find an existing MCP that might handle this command
        for mcp_name, mcp_data in self.mcp_box.mcps.items():
            logger.debug(f"Checking if MCP '{mcp_name}' can handle command: '{command_str}'")
            if mcp_data.get('original_command') == command_str:
                # Exact match found, reuse this MCP
                logger.info(f"Found existing MCP '{mcp_name}' that matches command exactly")
                try:
                    logger.info(f"Executing MCP '{mcp_name}'")
                    output = mcp_data["function"]()
                    if inspect.isgenerator(output):
                        logger.debug(f"MCP '{mcp_name}' returned a generator")
                        for chunk in output:
                            logger.debug(f"MCP output chunk: {chunk[:100]}...")
                            yield chunk
                    else:
                        logger.debug(f"MCP '{mcp_name}' returned: {str(output)[:100]}...")
                        yield output
                    return
                except Exception as e:
                    error_msg = f"Error executing existing MCP '{mcp_name}': {e}"
                    logger.error(error_msg, exc_info=True)
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    yield error_msg
                    return

        # Before creating MCPs, check if we can handle this with web search
        logger.info("Checking if query can be handled via web search")
        yield "🔍 Checking if this can be answered with web search...\n"
        
        if self.web_agent.can_handle_with_search(command_str):
            logger.info("Query appears suitable for web search")
            yield "🌐 This looks like a search query. Searching the web...\n\n"
            
            try:
                logger.info(f"Attempting web search for: '{command_str}'")
                web_answer = self.web_agent.answer_query(command_str)
                if web_answer:
                    logger.info("Web search returned results")
                    yield web_answer
                    return
                else:
                    logger.info("Web search returned no useful results")
                    yield "⚠️ Web search didn't return useful results. Creating a custom tool instead...\n\n"
            except Exception as e:
                logger.error(f"Web search failed: {str(e)}", exc_info=True)
                yield f"⚠️ Web search failed ({str(e)}). Creating a custom tool instead...\n\n"
        else:
            logger.info("Query not suitable for web search, proceeding to MCP creation")
            yield "🔧 This doesn't look like a search query. Creating a custom tool...\n\n"

        # Check if we can reuse an existing MCP based on intent
        yield "🤖 Analyzing command to find or create an appropriate tool...\n"
        
        # First, check if any existing MCP can handle this command by testing them
        suitable_existing_mcp = self._find_suitable_existing_mcp(command_str)
        
        if suitable_existing_mcp:
            mcp_name, existing_mcp = suitable_existing_mcp
            logger.info(f"Found suitable existing MCP '{mcp_name}' that can handle this command")
            yield f"✅ Found existing '{mcp_name}' tool that can handle this request - reusing it\n"
            
            # Execute the existing MCP directly
            yield "\n" + "─" * 30 + "\n"
            
            try:
                # Set the current command context for the parameterized function
                import builtins
                original_builtin_command = getattr(builtins, '_current_user_command', None)
                builtins._current_user_command = command_str
                
                try:
                    output = existing_mcp["function"]()
                finally:
                    # Restore the original command context
                    if original_builtin_command is not None:
                        builtins._current_user_command = original_builtin_command
                    else:
                        if hasattr(builtins, '_current_user_command'):
                            delattr(builtins, '_current_user_command')
                
                if output is None:
                    error_msg = f"Error: MCP '{mcp_name}' returned None. MCPs must return a string value."
                    logger.error(error_msg)
                    yield error_msg
                    return
                    
                logger.debug(f"Reused MCP '{mcp_name}' returned: {str(output)[:100]}...")
                yield output
                return
                
            except Exception as e:
                logger.warning(f"Existing MCP '{mcp_name}' failed to handle command: {e}")
                yield f"⚠️ Existing tool failed, creating a new one...\n"
                # Fall through to create a new MCP
        
        # Generate MCP name to see if we already have a suitable tool
        tokens = self._parse_command_tokens(command_str)
        potential_mcp_name = self._generate_smart_mcp_name(command_str, tokens)
        
        # Check if we already have this type of MCP
        if potential_mcp_name in self.mcp_box.mcps:
            existing_mcp = self.mcp_box.mcps[potential_mcp_name]
            logger.info(f"Found existing parameterized MCP '{potential_mcp_name}' - reusing it")
            yield f"✅ Found existing '{potential_mcp_name}' tool - reusing it for this request\n"
            
            # Execute the existing MCP directly
            yield "\n" + "─" * 30 + "\n"
            
            try:
                # Set the current command context for the parameterized function
                import builtins
                original_builtin_command = getattr(builtins, '_current_user_command', None)
                builtins._current_user_command = command_str
                
                try:
                    output = existing_mcp["function"]()
                finally:
                    # Restore the original command context
                    if original_builtin_command is not None:
                        builtins._current_user_command = original_builtin_command
                    else:
                        if hasattr(builtins, '_current_user_command'):
                            delattr(builtins, '_current_user_command')
                
                if output is None:
                    error_msg = f"Error: MCP '{potential_mcp_name}' returned None. MCPs must return a string value."
                    logger.error(error_msg)
                    yield error_msg
                    return
                    
                logger.debug(f"Reused MCP '{potential_mcp_name}' returned: {str(output)[:100]}...")
                yield output
                return
                
            except Exception as e:
                error_msg = f"Error executing reused MCP '{potential_mcp_name}': {e}"
                logger.error(error_msg, exc_info=True)
                yield error_msg
                return

        # No existing MCP found, create a new one
        mcp_data = None
        mcp_display_name = "new_tool"
        
        logger.info("No suitable existing MCP found, creating new one")
        yield f"🧠 Creating a new '{potential_mcp_name}' tool...\n\n"
        
        for stream_item in self._attempt_dynamic_mcp_creation_streaming(command_str, potential_mcp_name):
            # Check if this is the special signal that MCP was created
            if isinstance(stream_item, tuple) and stream_item[0] == "__MCP_CREATED__":
                mcp_data = stream_item[1]
                mcp_display_name = stream_item[2]
                logger.info(f"Successfully created new MCP: '{mcp_display_name}'")
                break
            else:
                # Stream the LLM generation process
                logger.debug(f"MCP creation progress: {str(stream_item)[:50]}...")
                yield stream_item
        
        if not mcp_data:
            error_msg = f"Sorry, I couldn't create a new tool to handle '{command_str}' at this time."
            logger.error(error_msg)
            yield error_msg
            return

        # Execute the newly created MCP
        try:
            logger.info(f"Executing newly created MCP '{mcp_display_name}'")
            
            # Add a separator line before executing the MCP
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
                error_msg = f"Error: MCP '{mcp_display_name}' returned None. MCPs must return a string value."
                logger.error(error_msg)
                yield error_msg
                return
                
            logger.debug(f"New MCP '{mcp_display_name}' returned: {str(output)[:100]}...")
            yield output
        except Exception as e:
            error_msg = f"Error executing new MCP '{mcp_display_name}': {e}"
            logger.error(error_msg, exc_info=True)
            logger.error(f"Stack trace: {traceback.format_exc()}")
            yield error_msg

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
            
            # Check if we already have an MCP with this name that could be reused
            if mcp_name in self.mcp_box.mcps:
                logger.info(f"Found existing MCP '{mcp_name}' that could handle this type of command")
                # Check if the existing MCP is parameterized and can handle the new command
                existing_mcp = self.mcp_box.mcps[mcp_name]
                
                # For now, let's reuse existing parameterized MCPs
                # In the future, we could add more sophisticated checks here
                logger.info(f"Reusing existing parameterized MCP '{mcp_name}' for new command")
                yield (f"__MCP_REUSED__", existing_mcp, mcp_name)
                return
            
            # Generate MCP script using LLM
            logger.info("Generating MCP script using LLM")
            yield f"🧠 Creating a reusable '{mcp_name}' tool...\n\n"
            
            for progress_chunk in self.llm_provider.generate_mcp_script_streaming(mcp_name, command_str):
                yield progress_chunk
            
            final_script = self.llm_provider.get_last_generated_mcp_script()
            
            if not final_script:
                logger.error("Failed to generate MCP script: Empty result returned")
                yield "Failed to generate a tool script. Please try again with a more specific command."
                return
            
            # Create MCP function from the generated script
            logger.info("Creating MCP function from generated script")
            mcp_function, mcp_metadata = self.mcp_factory.create_mcp_from_script(mcp_name, final_script)
            
            if not mcp_function:
                logger.error("Failed to create MCP function from script")
                yield "Failed to create a working tool. The generated code had issues. Please try again."
                return
            
            # Debug - Log the actual function code
            logger.debug(f"Created MCP function: {mcp_function.__code__}")
            
            # Get display name from metadata or use the generated name
            mcp_display_name = mcp_metadata.get('name', mcp_name)
            logger.info(f"Using display name '{mcp_display_name}' for new MCP")
            
            # Register the MCP
            logger.info(f"Registering new MCP '{mcp_display_name}'")
            mcp_metadata['original_command'] = command_str
            self.mcp_box.register_mcp(
                name=mcp_display_name,
                function=mcp_function,
                metadata=mcp_metadata
            )
            
            # Return a special signal that MCP was created, along with the MCP data
            mcp_data = self.mcp_box.mcps[mcp_display_name]
            logger.info(f"Successfully created and registered MCP '{mcp_display_name}'")
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
            logger.warning(f"Error parsing command tokens for '{command_str}', falling back to simple split")
            return command_str.split()

    def _generate_smart_mcp_name(self, command_str: str, tokens: List[str]) -> str:
        """Use LLM to generate a smart, reusable MCP name based on command intent."""
        try:
            # Use LLM to analyze the command and generate an appropriate MCP name
            prompt = f"""Analyze this user command and generate a generic, reusable MCP (Micro Control Program) name.

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

    def _find_suitable_existing_mcp(self, command_str: str) -> Optional[Tuple[str, Dict]]:
        """Find an existing MCP that can handle the given command based on intent."""
        logger.info(f"Finding suitable existing MCP for command: '{command_str}'")
        
        # Check for GitHub stars queries specifically
        if 'stars' in command_str.lower() and ('github' in command_str.lower() or '/' in command_str):
            # Look for any existing GitHub-related MCP
            github_mcps = [name for name in self.mcp_box.mcps.keys() 
                          if 'github' in name.lower() and 'star' in name.lower()]
            
            if github_mcps:
                mcp_name = github_mcps[0]  # Use the first GitHub stars MCP found
                existing_mcp = self.mcp_box.mcps[mcp_name]
                
                # Check if this MCP is hardcoded by testing it with a different repo
                # If it's truly parameterized, it should handle different repos correctly
                if self._is_mcp_hardcoded(existing_mcp, command_str):
                    logger.warning(f"Existing MCP '{mcp_name}' appears to be hardcoded, removing it")
                    del self.mcp_box.mcps[mcp_name]
                    return None
                
                logger.info(f"Found existing GitHub stars MCP: '{mcp_name}'")
                return mcp_name, existing_mcp
        
        # Check for weather queries
        if 'weather' in command_str.lower():
            weather_mcps = [name for name in self.mcp_box.mcps.keys() 
                           if 'weather' in name.lower()]
            if weather_mcps:
                mcp_name = weather_mcps[0]
                logger.info(f"Found existing weather MCP: '{mcp_name}'")
                return mcp_name, self.mcp_box.mcps[mcp_name]
        
        # Check for greeting commands
        if any(word in command_str.lower() for word in ['greet', 'hello', 'hi']):
            greeting_mcps = [name for name in self.mcp_box.mcps.keys() 
                            if any(word in name.lower() for word in ['greet', 'hello'])]
            if greeting_mcps:
                mcp_name = greeting_mcps[0]
                logger.info(f"Found existing greeting MCP: '{mcp_name}'")
                return mcp_name, self.mcp_box.mcps[mcp_name]
        
        # Check for calculation commands
        if any(word in command_str.lower() for word in ['calculate', 'area', 'circle', 'square']):
            calc_mcps = [name for name in self.mcp_box.mcps.keys() 
                        if any(word in name.lower() for word in ['calculate', 'area', 'math'])]
            if calc_mcps:
                mcp_name = calc_mcps[0]
                logger.info(f"Found existing calculation MCP: '{mcp_name}'")
                return mcp_name, self.mcp_box.mcps[mcp_name]
        
        # Check for currency conversion
        if any(word in command_str.lower() for word in ['convert', 'currency', 'usd', 'eur', 'exchange']):
            currency_mcps = [name for name in self.mcp_box.mcps.keys() 
                            if any(word in name.lower() for word in ['convert', 'currency', 'exchange'])]
            if currency_mcps:
                mcp_name = currency_mcps[0]
                logger.info(f"Found existing currency MCP: '{mcp_name}'")
                return mcp_name, self.mcp_box.mcps[mcp_name]
        
        # Check for time/date queries
        if any(word in command_str.lower() for word in ['time', 'date', 'clock']):
            time_mcps = [name for name in self.mcp_box.mcps.keys() 
                        if any(word in name.lower() for word in ['time', 'date', 'clock'])]
            if time_mcps:
                mcp_name = time_mcps[0]
                logger.info(f"Found existing time MCP: '{mcp_name}'")
                return mcp_name, self.mcp_box.mcps[mcp_name]
        
        logger.info("No suitable existing MCP found based on intent matching")
        return None

    def _is_mcp_hardcoded(self, mcp_data: Dict, command_str: str) -> bool:
        """Check if an MCP is hardcoded by looking at its source code."""
        try:
            # Try to inspect the function source code to detect hardcoded values
            function = mcp_data.get('function')
            if not function:
                return False
            
            # Get the function's source code if possible
            try:
                import inspect
                source = inspect.getsource(function)
                
                # Look for hardcoded command strings or repository names
                hardcoded_patterns = [
                    r'command\s*=\s*["\'].*?["\']',  # command = "hardcoded string"
                    r'repo_path\s*=\s*["\'].*?["\']',  # repo_path = "hardcoded/repo"
                    r'ryantzr1',  # specific username
                    r'OpenAlita',  # specific repo name
                ]
                
                for pattern in hardcoded_patterns:
                    if re.search(pattern, source, re.IGNORECASE):
                        logger.info(f"MCP appears hardcoded - found pattern: {pattern}")
                        return True
                        
                # Check if the function doesn't use builtins._current_user_command
                if 'builtins' not in source or '_current_user_command' not in source:
                    logger.info("MCP appears hardcoded - doesn't use dynamic command access")
                    return True
                    
            except (OSError, TypeError):
                # Can't get source code, assume it's okay
                logger.debug("Could not inspect MCP source code")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking if MCP is hardcoded: {e}")
            return False
        
        return False

    async def process_command_async(self, command_str: str) -> str:
        """Async version of command processing that can use LangGraph workflows."""
        if not command_str.strip():
            return "Please enter a command."

        logger.info(f"Processing command async: '{command_str}'")

        # Check if this command should use the LangGraph workflow
        if self.langgraph_manager.should_use_langgraph(command_str):
            logger.info("Command routed to LangGraph multi-agent workflow")
            return await self._process_with_langgraph(command_str)
        else:
            logger.info("Command routed to traditional single-agent processing")
            return await self._process_with_traditional_agent(command_str)

    async def _process_with_langgraph(self, command_str: str) -> str:
        """Process command using LangGraph multi-agent workflow."""
        try:
            logger.info(f"LangGraph processing: {command_str}")
            
            # Use the LangGraph workflow manager
            output_messages = await self.langgraph_manager.process_query_streaming(command_str)
            
            if output_messages:
                # Combine all agent messages into a cohesive response
                response_parts = []
                response_parts.append("🔄 **Multi-Agent Workflow Results:**\n")
                
                for i, message in enumerate(output_messages[:-1]):  # All but the last message
                    response_parts.append(f"{message}\n")
                
                # Add final response prominently
                if output_messages:
                    response_parts.append("\n" + "="*50)
                    response_parts.append(f"\n**Final Answer:**\n{output_messages[-1]}")
                
                return "\n".join(response_parts)
            else:
                return "I apologize, but I couldn't process your request through the multi-agent workflow."
                
        except Exception as e:
            logger.error(f"Error in LangGraph processing: {e}", exc_info=True)
            return f"Error in multi-agent workflow: {str(e)}"

    async def _process_with_traditional_agent(self, command_str: str) -> str:
        """Process command using traditional single-agent approach."""
        try:
            # Convert streaming response to single response
            result_parts = []
            for chunk in self.process_command_streaming(command_str):
                if chunk != "quit_signal":
                    result_parts.append(chunk)
            
            return "".join(result_parts)
            
        except Exception as e:
            logger.error(f"Error in traditional processing: {e}", exc_info=True)
            return f"Error processing command: {str(e)}"

    def process_command_with_routing(self, command_str: str) -> Generator[str, None, None]:
        """Enhanced command processing with intelligent routing between single and multi-agent workflows."""
        if not command_str.strip():
            yield "Please enter a command."
            return

        logger.info(f"Processing command with intelligent routing: '{command_str}'")

        # Quick commands that don't need LangGraph
        if command_str.lower().strip() in ["help", "quit"]:
            for chunk in self.process_command_streaming(command_str):
                yield chunk
            return

        # Determine if this should use LangGraph
        if self.langgraph_manager.should_use_langgraph(command_str):
            yield "🔄 **Routing to Multi-Agent Workflow**\n"
            yield "This query will be processed by multiple specialized agents working together...\n\n"
            
            # Run async LangGraph processing
            try:
                import asyncio
                
                # Check if we're already in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, need to handle this differently
                    result = asyncio.run_coroutine_threadsafe(
                        self._process_with_langgraph(command_str), 
                        loop
                    ).result(timeout=60)
                    yield result
                except RuntimeError:
                    # No running loop, we can use asyncio.run
                    result = asyncio.run(self._process_with_langgraph(command_str))
                    yield result
                    
            except Exception as e:
                logger.error(f"Error in LangGraph workflow: {e}", exc_info=True)
                yield f"⚠️ Multi-agent workflow failed: {str(e)}\n"
                yield "Falling back to traditional processing...\n\n"
                
                # Fallback to traditional processing
                for chunk in self.process_command_streaming(command_str):
                    yield chunk
        else:
            yield "🔧 **Using Traditional Agent Processing**\n"
            yield "This query will be handled by the core Alita agent...\n\n"
            
            # Use traditional streaming processing
            for chunk in self.process_command_streaming(command_str):
                yield chunk

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get status information about the workflow systems."""
        return {
            "langgraph_enabled": self.langgraph_manager is not None,
            "available_agents": [
                "orchestrator",
                "researcher", 
                "analyzer",
                "synthesizer",
                "executor"
            ],
            "traditional_agent_mcps": len(self.mcp_box.mcps),
            "web_agent_enabled": self.web_agent is not None
        }

    def explain_workflow_routing(self, command_str: str) -> str:
        """Explain why a command would be routed to a particular workflow."""
        uses_langgraph = self.langgraph_manager.should_use_langgraph(command_str)
        
        explanation = f"**Workflow Routing Analysis for:** '{command_str}'\n\n"
        
        if uses_langgraph:
            explanation += "✅ **Will use Multi-Agent LangGraph Workflow**\n\n"
            explanation += "**Reasons:**\n"
            
            query_lower = command_str.lower()
            
            # Check for research indicators
            research_indicators = [
                'analyze', 'compare', 'research', 'investigate', 'study',
                'what is', 'how does', 'why', 'explain', 'tell me about'
            ]
            found_research = [ind for ind in research_indicators if ind in query_lower]
            if found_research:
                explanation += f"- Contains research keywords: {', '.join(found_research)}\n"
            
            # Check for multi-step indicators
            multi_step_indicators = [
                'first', 'then', 'after', 'next', 'finally',
                'step by step', 'process', 'workflow'
            ]
            found_multi_step = [ind for ind in multi_step_indicators if ind in query_lower]
            if found_multi_step:
                explanation += f"- Contains multi-step keywords: {', '.join(found_multi_step)}\n"
            
            # Check for analysis indicators
            analysis_indicators = [
                'statistics', 'trends', 'patterns', 'insights',
                'correlation', 'relationship', 'impact'
            ]
            found_analysis = [ind for ind in analysis_indicators if ind in query_lower]
            if found_analysis:
                explanation += f"- Contains analysis keywords: {', '.join(found_analysis)}\n"
            
            if len(command_str.split()) > 8:
                explanation += f"- Query is complex ({len(command_str.split())} words)\n"
            
            explanation += "\n**Workflow Steps:**\n"
            explanation += "1. 🤖 Orchestrator will analyze task complexity\n"
            explanation += "2. 🔍 Researcher may gather information from web sources\n"
            explanation += "3. 📊 Analyzer may extract insights from gathered data\n"
            explanation += "4. ⚙️ Executor may create/run specialized tools\n"
            explanation += "5. 🎯 Synthesizer will create the final comprehensive response\n"
            
        else:
            explanation += "⚡ **Will use Traditional Single-Agent Processing**\n\n"
            explanation += "**Reasons:**\n"
            explanation += "- Query is straightforward and doesn't require multi-agent collaboration\n"
            explanation += "- Can be handled efficiently by core Alita agent\n"
            explanation += "- May use web search or create/execute MCPs as needed\n"
        
        return explanation