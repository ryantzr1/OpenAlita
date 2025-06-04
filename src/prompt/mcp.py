import shlex
import re # For parsing LLM output
import inspect # Added for checking if MCP output is a generator
import os
from dotenv import load_dotenv

# --- Prompts (can be expanded) ---
MULTIMEDIA_RESPONSE_PROMPT = "The tool '{tool_name}' has returned a multimedia response."
SYSTEM_PROMPT = "You are interacting with Alita. How can I assist you?"
NEXT_STEP_PROMPT = "What would you like to do next? (Type 'quit' to exit)"

# --- LLM Interaction (Placeholder for DeepSeek API or other LLMs) ---
class LLMProvider:
    """
    Placeholder for interacting with an LLM API like DeepSeek.
    You'll need to replace the '_make_api_call' method with actual API calls.
    """
    def __init__(self, api_key=None, api_url=None):
        # --- TODO: REPLACE WITH YOUR ACTUAL API KEY AND ENDPOINT ---
        load_dotenv() # Load environment variables from .env file

        self.api_key = api_key if api_key else os.getenv("ANTHROPIC_API_KEY")
        self.api_url = api_url if api_url else "https://api.anthropic.com/v1/"
        # Example for DeepSeek: self.model_name = "deepseek-coder"
        # Example for OpenAI: self.model_name = "gpt-3.5-turbo"
        self.model_name = "claude-3-5-sonnet-20241022"  # Change to your desired model

    def generate_mcp_script(self, mcp_name, task_description, args_details, user_command):
        """
        Generate an MCP script using Claude API based on the provided parameters.
        
        Args:
            mcp_name (str): Name of the MCP function to generate
            task_description (str): Description of what the MCP should do
            args_details (str): Details about the arguments the MCP should accept
            user_command (str): Original user command that triggered this generation
            
        Returns:
            str: Generated MCP script as a string, or None if generation failed
        """
        prompt = self._build_mcp_generation_prompt(
            mcp_name, task_description, args_details, user_command
        )
        
        script_chunks = []
        first_chunk_processed = False
        try:
            for chunk in self._make_api_call(prompt):
                if not first_chunk_processed:
                    if isinstance(chunk, str) and chunk.startswith("Error:"):
                        print(f"   LLM Log: API call failed: {chunk}")
                        return None
                    first_chunk_processed = True
                script_chunks.append(chunk)
            
            if not script_chunks: # Handle case where stream is empty but no explicit error was yielded
                print(f"   LLM Log: No script content received from API for {mcp_name}.")
                return None

            return "".join(script_chunks)
        except Exception as e:
            print(f"   LLM Log: Exception while collecting script chunks for {mcp_name}: {e}")
            return None
    
    def _get_time_mcp_specific_guidance(self) -> str:
        return (
            "\n\n5. **CRITICAL INSTRUCTIONS FOR 'time' MCP:**\n"
            "   - You *MUST* import the `datetime` module using `import datetime`.\n"
            "   - To get the current time object, you *MUST* use `datetime.datetime.now()`.\n"
            "   - Do *NOT* use `datetime.now()` directly on the imported module; it will fail, as `datetime` refers to the module, not the class.\n"
            "   - Format the time using the `strftime` method on the time object (e.g., `current_time_object.strftime('%%H:%%M:%%S')`).\n"
            "   - **Correct Example Snippet:**\n"
            "     ```python\n"
            "     import datetime\n"
            "     # ... other parts of your function ...\n"
            "     try:\n"
            "         current_time_object = datetime.datetime.now()\n"
            "         formatted_time_string = current_time_object.strftime('%%H:%%M:%%S %%p') # Example format\n"
            "         return formatted_time_string\n"
            "     except Exception as e:\n"
            "         return f\"Error processing time: {{e}}\"\n"
            "     ```"
        )

    def _build_mcp_generation_prompt(self, mcp_name, task_description, args_details, user_command):
        """
        Build a comprehensive prompt for MCP script generation.
        
        Args:
            mcp_name (str): Name of the MCP function
            task_description (str): What the MCP should do
            args_details (str): Argument specifications
            user_command (str): Original user command
            
        Returns:
            str: Complete prompt for the LLM
        """
        time_specific_guidance_text = ""
        if mcp_name == "time":
            time_specific_guidance_text = self._get_time_mcp_specific_guidance()

        prompt = f"""You are an expert Python developer tasked with creating a Micro Control Program (MCP) function.

MCP Name to define: {mcp_name}

Task Description: {task_description}

Arguments Details: {args_details}

Original User Command: {user_command}

Please generate a complete, functional Python MCP script that:

1. **Follows this exact format structure:**
   ```
   # MCP Name: {mcp_name}
   # Description: [clear description of what this MCP does]
   # Arguments: [list each argument with type and description]
   # Returns: [what the function returns]
   # Requires: [any imports needed, if applicable]
   
   [import statements if needed]
   
   def {mcp_name}([parameters]):
       [implementation]
   ```

2. **Requirements:**
   - The function must be named exactly `{mcp_name}`
   - Include comprehensive error handling with try/catch blocks
   - Return meaningful error messages as strings when things go wrong
   - Use proper type hints where appropriate
   - Handle edge cases gracefully
   - Include clear comments explaining the logic
   - If external libraries are needed, import them at the top

3. **Best Practices:**
   - Validate all inputs
   - Handle common error scenarios (invalid input, network issues, file not found, etc.)
   - Return consistent data types
   - Use descriptive variable names
   - Keep the function focused on a single responsibility
   - When using the 'datetime' module for current time, ensure you use 'datetime.datetime.now()'.

4. **Error Handling Pattern:**
   ```python
   try:
       # main logic here
       return result
   except SpecificException as e:
       return f"Error: {{e}}"
   except Exception as e:
       return f"Unexpected error in {mcp_name}: {{e}}"
   ```{time_specific_guidance_text}

Generate ONLY the MCP script with proper formatting. Do not include any explanatory text before or after the script."""

        return prompt
    def _make_api_call(self, prompt_text):
        """
        Makes an HTTP request to Anthropic's Claude API with streaming enabled.
        
        Args:
            prompt_text (str): The prompt to send to the LLM
            
        Returns:
            generator: Yields text chunks as they arrive from Claude's API, or an error string.
        """
        import requests
        import json
        
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model_name,
            "max_tokens": 1500,
            "temperature": 0.5,
            "stream": True,
            "messages": [{"role": "user", "content": prompt_text}]
        }
        
        try:
            print(f"   LLM Log: Making streaming API call to Anthropic Claude (Model: {self.model_name})...")
            response = requests.post(url, headers=headers, json=payload, timeout=20, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    line_data_str = line[6:].decode('utf-8')
                    if line_data_str == "[DONE]": # Some APIs might use [DONE] marker
                        print("   LLM Log: Received [DONE] marker.")
                        break
                    try:
                        data = json.loads(line_data_str)
                        event_type = data.get("type")

                        if event_type == "content_block_delta":
                            delta_text = data.get("delta", {}).get("text")
                            if delta_text:
                                print(f"   LLM Log: Yielding chunk: '{delta_text}'")
                                yield delta_text
                        elif event_type == "message_start":
                            print(f"   LLM Log: Stream - Message start event received. Input tokens: {data.get('message',{}).get('usage',{}).get('input_tokens')}")
                        elif event_type == "message_delta":
                            # Contains usage updates, stop_reason
                            usage = data.get("usage", {})
                            if usage and usage.get("output_tokens"):
                                print(f"   LLM Log: Stream - Message delta event. Output tokens so far: {usage['output_tokens']}")
                            stop_reason = data.get("delta", {}).get("stop_reason")
                            if stop_reason:
                                print(f"   LLM Log: Stream - Message delta. Stop Reason: {stop_reason}")
                        elif event_type == "message_stop":
                            print("   LLM Log: Stream - Message stop event received.")
                            break # End of message
                        elif event_type == "error":
                            error_message = data.get("error", {}).get("message", "Unknown API error from stream")
                            print(f"   LLM Log: Error event from Anthropic stream: {error_message}")
                            yield f"Error: API Stream Error - {error_message}"
                            return # Stop processing on stream error
                        # else: # Other event types like content_block_start, content_block_stop, ping
                            # print(f"   LLM Log: Skipping event type: {event_type}")

                    except json.JSONDecodeError:
                        print(f"   LLM Log: Skipping non-JSON line in stream: {line_data_str}")
                        continue
            
            print(f"   LLM Log: Finished processing Anthropic stream.")
            
        except requests.exceptions.Timeout:
            print(f"   LLM Log: Timeout calling Anthropic API after 20 seconds.")
            yield "Error: API timeout"
        except requests.exceptions.RequestException as e:
            print(f"   LLM Log: HTTP error calling Anthropic API: {e}")
            error_text = str(e)
            if hasattr(e, 'response') and e.response is not None and e.response.text:
                error_text = f"{str(e)} - {e.response.text[:500]}" # Add response details
            yield f"Error: API request failed - {error_text}"
        except Exception as e:
            print(f"   LLM Log: Unexpected error in _make_api_call: {e}")
            yield f"Error: Unexpected API error - {str(e)}"

class MCPBox:
    def __init__(self):
        self.mcps = {} # name -> {function, description, args_info, returns_info, source, requires, script_content}

    def add_mcp(self, name: str, function, description: str, args_info: str = "N/A", returns_info: str = "N/A", source: str = "pre-loaded", requires: str = None, script_content: str = None):
        if name not in self.mcps:
            self.mcps[name] = {
                "function": function,
                "description": description,
                "args_info": args_info,
                "returns_info": returns_info,
                "source": source, # "pre-loaded", "dynamically-generated"
                "requires": requires,
                "script_content": script_content if source == "dynamically-generated" else None # Store script only for dynamic MCPs
            }
            print(f"   Agent Log: New MCP '{name}' ({source}) added. Description: {description}")
        else:
            print(f"   Agent Log: MCP '{name}' already exists. Not overriding.")

    def get_mcp(self, name: str):
        return self.mcps.get(name)

    def list_mcps(self):
        if not self.mcps:
            return "MCP Box is empty."
        mcp_list = "Available MCPs:\n"
        for name, data in self.mcps.items():
            mcp_list += f"- {name} (Source: {data['source']})\n"
            mcp_list += f"    Description: {data['description']}\n"
            mcp_list += f"    Arguments: {data['args_info']}\n"
            mcp_list += f"    Returns: {data['returns_info']}\n"
            if data['requires']:
                mcp_list += f"    Requires: {data['requires']}\n"
            if data.get('script_content'):
                mcp_list += f"    --- Script Start ---\n{data['script_content'][:200]}...\n    --- Script End ---\n" # Show a snippet in help
        return mcp_list
    

class MCPFactory:
    """
    Responsible for taking a script string (from LLM) and loading it.
    WARNING: The use of exec() here is for demonstration and is NOT SAFE without sandboxing.
    """
    def _parse_mcp_metadata_and_code(self, script_string: str):
        metadata = {
            "name": None,
            "description": "No description provided by LLM.",
            "args": "N/A",
            "returns": "N/A",
            "requires": None,
            "code_body": script_string # Default to full script if parsing fails
        }
        
        # First, clean up the script string by removing Markdown code block markers if present
        lines = script_string.splitlines()
        if lines and lines[0].strip().startswith("```"):
            # Remove the opening code block marker
            lines.pop(0)
            # Remove the closing code block marker if it exists
            if lines and lines[-1].strip() == "```":
                lines.pop()
        
        script_string = "\n".join(lines)
        
        header_lines = []
        code_body_lines = []
        in_header = True # Flag to know if we are currently parsing header lines

        # Iterate through each line of the script string
        for line_num, line in enumerate(script_string.splitlines()):
            stripped_line = line.strip()
            if in_header:
                if stripped_line.startswith("#"):
                    header_lines.append(stripped_line)
                # If we encounter a line that doesn't start with # and is not empty, header ends.
                elif stripped_line and not stripped_line.startswith("#"):
                    in_header = False
                    code_body_lines.append(line) # This line is part of the code body
                elif not stripped_line: # Allow empty lines in header but continue
                    header_lines.append(line) 
                    
            else: # We are in the code body part
                code_body_lines.append(line)
        
        metadata["code_body"] = "\n".join(code_body_lines).strip()

        # Post-process code_body to remove erroneous `from typing import str`
        if "from typing import str" in metadata["code_body"]:
            code_lines = metadata["code_body"].splitlines()
            cleaned_code_lines = [line for line in code_lines if line.strip() != "from typing import str"]
            metadata["code_body"] = "\n".join(cleaned_code_lines)

        for line in header_lines:
            if line.startswith("# MCP Name:"):
                metadata["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Description:"):
                metadata["description"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Arguments:"):
                metadata["args"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Returns:"):
                metadata["returns"] = line.split(":", 1)[1].strip()
            elif line.startswith("# Requires:"): # Can be multiple, accumulate or take first for simplicity here
                if metadata["requires"] is None: metadata["requires"] = []
                metadata["requires"].append(line.split(":", 1)[1].strip())
        
        if isinstance(metadata["requires"], list): # Join if multiple
            metadata["requires"] = ", ".join(metadata["requires"])

        return metadata

    def create_mcp_from_script(self, mcp_name_expected: str, script_string: str):
        if not script_string or not script_string.strip():
            print("   MCPFactory Log: No script string provided by LLM.")
            return None, {}

        metadata = self._parse_mcp_metadata_and_code(script_string)
        actual_mcp_name = metadata.get("name") or mcp_name_expected # Fallback to expected name
        
        if not metadata["code_body"]:
            print(f"   MCPFactory Log: No executable code body found for MCP '{actual_mcp_name}'.")
            print(f"--- Received Script --- \n{script_string}\n --- End Script ---")
            return None, metadata

        # --- !!! WARNING: EXTREME SECURITY RISK !!! ---
        # Executing arbitrary code from any source, especially an LLM, is highly dangerous.
        # This can lead to severe security vulnerabilities on your system.
        # In a real system, this MUST happen in a SECURE, ISOLATED SANDBOXED ENVIRONMENT.
        # The 'exec()' function used here is for basic demonstration purposes ONLY.
        # --- !!! DO NOT USE THIS IN PRODUCTION OR WITH UNTRUSTED LLMS/CODE WITHOUT A PROPER SANDBOX !!! ---
        try:
            print(f"   MCPFactory Log: Attempting to load MCP '{actual_mcp_name}' from script.")
            
            # Create a comprehensive restricted globals environment
            restricted_globals = {
                "__builtins__": {
                    # Basic types and functions
                    "print": print, "len": len, "str": str, "int": int, "float": float, "bool": bool,
                    "list": list, "dict": dict, "tuple": tuple, "set": set,
                    "True": True, "False": False, "None": None,
                    
                    # Math and utility functions
                    "abs": abs, "round": round, "range": range, "zip": zip, "map": map, "filter": filter,
                    "sum": sum, "min": min, "max": max, "pow": pow, "sorted": sorted,
                    
                    # Type checking and conversion
                    "isinstance": isinstance, "type": type, "hasattr": hasattr, "getattr": getattr,
                    "setattr": setattr, "delattr": delattr,
                    
                    # String and formatting
                    "format": format, "repr": repr, "ascii": ascii, "ord": ord, "chr": chr,
                    
                    # Iteration and enumeration
                    "enumerate": enumerate, "iter": iter, "next": next, "reversed": reversed,
                    
                    # Common exceptions
                    "ValueError": ValueError, "TypeError": TypeError, "Exception": Exception,
                    "AttributeError": AttributeError, "KeyError": KeyError, "IndexError": IndexError,
                    "ZeroDivisionError": ZeroDivisionError, "ImportError": ImportError,
                    
                    # Object utilities
                    "callable": callable, "vars": vars, "dir": dir,
                    
                    # Logical operations
                    "all": all, "any": any,

                    # Import functionality
                    "__import__": __import__
                }
            }
            
            # Add commonly required modules safely
            requires_str = metadata.get("requires", "").lower() if metadata.get("requires") else ""
            
            if "math" in requires_str:
                import math
                restricted_globals["math"] = math
                
            if "random" in requires_str:
                import random
                restricted_globals["random"] = random
                
            if "datetime" in requires_str:
                import datetime
                restricted_globals["datetime"] = datetime
                
            if "json" in requires_str:
                import json
                restricted_globals["json"] = json
                
            if "re" in requires_str:
                import re
                restricted_globals["re"] = re
                
            if "socket" in requires_str:
                import socket
                restricted_globals["socket"] = socket
                
            if "requests" in requires_str:
                try:
                    import requests
                    restricted_globals["requests"] = requests
                except ImportError:
                    print("   MCPFactory Log: Warning - requests module not available")
            
            if "time" in requires_str: # Added time module
                import time
                restricted_globals["time"] = time
                    
            if "os" in requires_str:
                import os
                # Create a safer os module with limited functions
                class SafeOS:
                    def __init__(self):
                        self.path = os.path
                        self.environ = dict(os.environ)  # Read-only copy
                    
                    def getenv(self, key, default=None):
                        return self.environ.get(key, default)
                        
                restricted_globals["os"] = SafeOS()
            
            # Add typing support
            if "typing" in requires_str or "Optional" in script_string:
                from typing import Optional, List, Dict, Any
                restricted_globals["Optional"] = Optional
                restricted_globals["List"] = List
                restricted_globals["Dict"] = Dict
                restricted_globals["Any"] = Any

            # Prepare the namespace for exec: start with our restricted_globals.
            # This dictionary will be used for both global and local variables during exec.
            mcp_namespace = dict(restricted_globals) 
            
            # Execute the script. Imports and definitions will populate mcp_namespace.
            exec(metadata['code_body'], mcp_namespace)

            if actual_mcp_name in mcp_namespace and callable(mcp_namespace[actual_mcp_name]):
                print(f"   MCPFactory Log: Successfully loaded function '{actual_mcp_name}'.")
                return mcp_namespace[actual_mcp_name], metadata
            else:
                print(f"   MCPFactory Log: Error: Function '{actual_mcp_name}' not found or not callable in generated script namespace.")
                print(f"   MCPFactory Log: Namespace after exec: {list(mcp_namespace.keys())}")
                print(f"--- Problematic Script for {actual_mcp_name} ---")
                print(script_string)
                print("--- End Problematic Script ---")
                return None, metadata
        except ImportError as e:
            imported_module_name = e.name
            required_modules_str = metadata.get("requires", "")
            required_modules_list = [m.strip().lower() for m in required_modules_str.split(',') if m.strip()]
            
            # Modules Alita explicitly handles in restricted_globals (names as they appear in # Requires)
            known_handled_modules = ["math", "random", "datetime", "json", "re", "socket", "requests", "os", "typing", "time"]

            if imported_module_name and imported_module_name.lower() in required_modules_list and imported_module_name.lower() not in known_handled_modules:
                print(f"   MCPFactory Log: ImportError for MCP '{actual_mcp_name}': Could not import '{imported_module_name}'.")
                print(f"   MCPFactory Log: This module was listed in # Requires, but is not in Alita's standard set of available modules.")
                print(f"   MCPFactory Log: Please ensure '{imported_module_name}' is installed in the environment if it's a custom/external library, or check if it can be replaced with a standard module.")
            else:
                print(f"   MCPFactory Log: CRITICAL ImportError executing generated script for '{actual_mcp_name}': {e}")
                print(f"--- Problematic Script for {actual_mcp_name} ---")
                print(script_string)
                print("--- End Problematic Script ---")
                return None, metadata
        except Exception as e:
            print(f"   MCPFactory Log: CRITICAL ERROR executing generated script for '{actual_mcp_name}': {e}")
            print(f"--- Problematic Script for {actual_mcp_name} ---")
            print(script_string)
            print("--- End Problematic Script ---")
            return None, metadata
        

class AlitaAgent:
    def __init__(self):
        self.mcp_box = MCPBox()
        self.llm_provider = LLMProvider()
        self.mcp_factory = MCPFactory()
        self._preload_basic_mcps()

    def _preload_basic_mcps(self):
        def mcp_add_numbers(a_str: str, b_str: str):
            try:
                return f"The sum is: {int(a_str) + int(b_str)}"
            except ValueError:
                return "Error: Please provide two numbers for addition."
        self.mcp_box.add_mcp(
            name="add", 
            function=mcp_add_numbers, 
            description="Adds two numbers provided as strings (e.g., add 5 3).", 
            args_info="a (number string), b (number string)", 
            returns_info="string (the sum or an error message)"
        )

        def mcp_generate_image_placeholder(description: str):
            tool_name = "SimpleImageGenerator"
            print(f"   Agent Log: {MULTIMEDIA_RESPONSE_PROMPT.format(tool_name=tool_name)}")
            return f"Placeholder: An image for '{description}' would be generated here by {tool_name}."
        self.mcp_box.add_mcp(
            name="image", 
            function=mcp_generate_image_placeholder, 
            description="Generates an image placeholder (e.g., image cat).", 
            args_info="description (string)", 
            returns_info="string (placeholder message)"
        )

        def mcp_echo_stream(text_to_echo: str):
            """
            # MCP Name: echo_stream
            # Description: Echoes the input text back, character by character with a delay.
            # Arguments: text_to_echo (str): The text to be echoed.
            # Returns: yields each character of the input string.
            # Requires: time
            """
            import time # This import is for the function's runtime, MCPFactory handles availability
            for char in text_to_echo:
                yield char
                # time.sleep(0.05) # Small delay to simulate streaming -- TEMPORARILY REMOVED FOR TESTING

        self.mcp_box.add_mcp(
            name="echo_stream",
            function=mcp_echo_stream,
            description="Echoes input text character by character (e.g., echo_stream Hello World).",
            args_info="text_to_echo (string)",
            returns_info="yields characters of the input string",
            requires="time" # Important: MCPFactory needs this to provide the 'time' module
        )

    def _attempt_dynamic_mcp_creation(self, action: str, args: list, original_command: str):
        print(f"AlitaAgent DEBUG: Entering _attempt_dynamic_mcp_creation for action: '{action}'")
        task_description = f"Handle the command '{action}'."
        if args:
            task_description += f" It was invoked with arguments: {', '.join(args)}."
        else:
            task_description += " It was invoked without arguments."
        
        args_details_for_llm = ""
        if not args:
            args_details_for_llm = "The function should accept no arguments."
        else:
            args_details_for_llm = f"The function should accept {len(args)} argument(s). These correspond to: {', '.join(args)}. Assume they are passed as strings."

        print(f"AlitaAgent DEBUG: Calling LLM provider to generate MCP script for '{action}'. Task: {task_description}")
        mcp_script_str = self.llm_provider.generate_mcp_script(
            mcp_name=action,
            task_description=task_description,
            args_details=args_details_for_llm,
            user_command=original_command
        )
        print(f"AlitaAgent DEBUG: LLM provider returned script (first 100 chars): '{mcp_script_str[:100] if mcp_script_str else 'None'}' for action '{action}'")

        if not mcp_script_str:
            print(f"   Agent Log: LLM failed to generate script for MCP '{action}'. (This is also an AlitaAgent DEBUG log)")
            return None, None

        print(f"AlitaAgent DEBUG: Calling MCP factory to create MCP from script for '{action}'")
        new_mcp_function, metadata = self.mcp_factory.create_mcp_from_script(action, mcp_script_str)
        print(f"AlitaAgent DEBUG: MCP factory returned function: '{bool(new_mcp_function)}' for action '{action}'")

        if new_mcp_function:
            mcp_actual_name = metadata.get("name") or action
            self.mcp_box.add_mcp(
                name=mcp_actual_name, 
                function=new_mcp_function,
                description=metadata.get("description", f"Dynamically generated MCP for '{mcp_actual_name}'."),
                args_info=metadata.get("args", "N/A"),
                returns_info=metadata.get("returns", "N/A"),
                source="dynamically-generated",
                requires=metadata.get("requires"),
                script_content=mcp_script_str # Pass the script content here
            )
            return self.mcp_box.get_mcp(mcp_actual_name), mcp_actual_name
        
        print(f"   Agent Log: Failed to create and load MCP '{action}' from the script generated by LLM. (This is also an AlitaAgent DEBUG log)")
        return None, None

    def _extract_intent_and_args(self, command_str: str):
        """
        Enhanced command parsing that can handle natural language queries
        and extract intents like 'weather' from 'what is the weather'
        """
        command_lower = command_str.lower().strip()
        
        # Define intent patterns
        weather_patterns = [
            r'\b(weather|temperature|forecast)\b',
            r'what.*weather',
            r'how.*weather',
            r'weather.*like'
        ]
        
        time_patterns = [
            r'\b(time|clock)\b',
            r'what.*time',
            r'current.*time',
            r'time.*now'
        ]
        
        # Check for weather intent
        for pattern in weather_patterns:
            if re.search(pattern, command_lower):
                return "weather", []
                
        # Check for time intent
        for pattern in time_patterns:
            if re.search(pattern, command_lower):
                return "time", []
        
        # Default parsing - try to split the command
        try:
            parts = shlex.split(command_str)
            if not parts:
                return None, []
        except ValueError:
            # Fallback to simple split if shlex fails
            parts = command_str.split()
            if not parts:
                return None, []
        
        action = parts[0].lower()
        args = parts[1:]
        
        # Special case: if the action is "time", ignore any arguments
        if action == "time":
            return "time", []
        
        return action, args
    
    def _process_command_internal(self, command_str: str):
        # Use enhanced parsing
        action, args = self._extract_intent_and_args(command_str)
        
        if not action:
            return "Please enter a command."

        if action == "help":
            return self.mcp_box.list_mcps()
        
        if action == "quit":
            return "quit_signal"

        mcp_data = self.mcp_box.get_mcp(action)

        if not mcp_data:
            mcp_data = self._attempt_dynamic_mcp_creation(action, args, command_str)
            if not mcp_data:
                return f"Sorry, I don't know how to '{action}', and I couldn't create a new tool for it at this time."

        mcp_function = mcp_data["function"]
        mcp_name = action
        
        try:
            print(f"   Agent Log: Executing MCP '{mcp_name}' with arguments: {args}")
            return mcp_function(*args)
        except TypeError as e:
            return f"Error executing MCP '{mcp_name}': {e}. Check arguments. Expected: {mcp_data.get('args_info', 'N/A')}"
        except Exception as e:
            return f"An unexpected error occurred within MCP '{mcp_name}': {e}"

    def process_command_streaming(self, command_str: str):
        print(f"AlitaAgent DEBUG: Entering process_command_streaming for command: '{command_str}'")
        action, args = self._extract_intent_and_args(command_str)

        if not action:
            print("AlitaAgent DEBUG: No action identified.")
            yield "Please enter a command."
            return

        if action == "help":
            print("AlitaAgent DEBUG: Processing 'help' command.")
            yield self.mcp_box.list_mcps()
            return
        
        if action == "quit":
            print("AlitaAgent DEBUG: Processing 'quit' command.")
            yield "quit_signal"
            return

        print(f"AlitaAgent DEBUG: Action '{action}' identified. Checking for existing MCP.")
        mcp_data = self.mcp_box.get_mcp(action)
        mcp_display_name = action 
        was_dynamically_created = False

        if not mcp_data:
            print(f"AlitaAgent DEBUG: No existing MCP for '{action}'. Attempting dynamic creation.")
            created_mcp_data, actual_name_used = self._attempt_dynamic_mcp_creation(action, args, command_str)
            
            if not created_mcp_data:
                print(f"AlitaAgent DEBUG: Dynamic MCP creation failed for '{action}'. Yielding error message.")
                yield f"Sorry, I don't know how to '{action}', and I couldn't create a new tool for it at this time."
                return
            
            mcp_data = created_mcp_data
            mcp_display_name = actual_name_used if actual_name_used else action
            was_dynamically_created = True
            print(f"AlitaAgent DEBUG: Dynamic MCP creation successful for '{mcp_display_name}'.")

        if was_dynamically_created:
            description = mcp_data.get("description", "No description provided.")
            args_info = mcp_data.get("args_info", "N/A")
            script_content = mcp_data.get("script_content", "No script content available.")
            
            yield f"Description: {description}\n"
            yield f"Arguments: {args_info}\n"
            
            if script_content != "No script content available.":
                yield "\n" # Add a blank line for better separation
                yield f"--- Generated Script for '{mcp_display_name}' ---\n"
                # Yield the script content, stripping any leading/trailing whitespace from the block
                # itself, then ensuring it ends with a single newline.
                yield f"{script_content.strip()}\n" 
                yield f"--- End Script ---\n"
                yield "\n" # Add another blank line for separation
            
            yield f"Now, I will try to execute it...\n---\n"

        mcp_function = mcp_data["function"]
        
        try:
            print(f"AlitaAgent DEBUG: Executing MCP '{mcp_display_name}' with arguments: {args}")
            output = mcp_function(*args) # Execute the MCP

            if inspect.isgenerator(output):
                print(f"AlitaAgent DEBUG: MCP '{mcp_display_name}' returned a generator. Iterating and yielding chunks.")
                for chunk in output:
                    yield chunk
                print(f"AlitaAgent DEBUG: Finished yielding chunks for generator MCP '{mcp_display_name}'.")
            else:
                # If the output is not a generator, it's a single result.
                print(f"AlitaAgent DEBUG: MCP '{mcp_display_name}' returned a single result (first 100 chars): '{str(output)[:100]}...'")
                yield output
                print(f"AlitaAgent DEBUG: Yielded single result for MCP '{mcp_display_name}'.")
        except TypeError as e:
            print(f"AlitaAgent DEBUG: TypeError executing MCP '{mcp_display_name}': {e}")
            yield f"Error executing MCP '{mcp_display_name}': {e}. Check arguments. Expected: {mcp_data.get('args_info', 'N/A')}"
        except Exception as e:
            print(f"AlitaAgent DEBUG: Exception executing MCP '{mcp_display_name}': {e}")
            yield f"An unexpected error occurred within MCP '{mcp_display_name}': {e}"

if __name__ == "__main__":
    print("Initializing Alita Agent (Self-Evolving Demo)...")
    agent = AlitaAgent()

    print(f"\n{SYSTEM_PROMPT}")
    print("Type 'help' for available commands, or 'quit' to exit.")
    print("Try commands like 'add 10 20', 'image a happy cat', or invent new ones like 'greet YourName' or 'circle_area 5'.")


    while True:
        try:
            user_input = input("\nAlita> ").strip()
            if not user_input:
                continue
            
            # Updated to handle streaming response
            full_response_chunks = []
            for chunk in agent.process_command_streaming(user_input):
                if chunk == "quit_signal":
                    print("Exiting Alita agent.")
                    exit() # Exit immediately
                
                # For CLI, print chunks as they arrive and accumulate
                print(chunk, end="", flush=True)
                full_response_chunks.append(str(chunk)) # Store chunk for any post-processing if needed
            
            print() # Newline after all chunks for a command are printed
            # final_response = "".join(full_response_chunks) # If you need the full response string later

        except EOFError: # Handle Ctrl+D
            print("\nExiting Alita agent.")
            break
        except KeyboardInterrupt: # Handle Ctrl+C
            print("\nExiting Alita agent.")
            break

    print("\nAlita Agent session ended.")