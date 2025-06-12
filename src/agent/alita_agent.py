import shlex
import re
import inspect

from .llm_provider import LLMProvider
from .mcp_box import MCPBox
from .mcp_factory import MCPFactory

MULTIMEDIA_RESPONSE_PROMPT = "The tool '{tool_name}' has returned a multimedia response."

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
            import time
            for char in text_to_echo:
                yield char

        self.mcp_box.add_mcp(
            name="echo_stream",
            function=mcp_echo_stream,
            description="Echoes input text character by character (e.g., echo_stream Hello World).",
            args_info="text_to_echo (string)",
            returns_info="yields characters of the input string",
            requires="time"
        )

    def _attempt_dynamic_mcp_creation(self, user_command: str):
        """Create a dynamic MCP based on the full user command description."""
        # Use the user's original command as the task description
        task_description = f"Handle the user request: '{user_command}'"
        
        # Generate a simple function name based on the command
        # This is just for internal naming, not parsing
        command_words = user_command.lower().split()
        if len(command_words) >= 2:
            function_name = "_".join(command_words[:2])  # e.g., "what_is" or "get_weather"
        elif command_words:
            function_name = command_words[0]  # e.g., "weather" or "time"
        else:
            function_name = "handle_request"
        
        # Clean the function name to be valid Python identifier
        import re
        function_name = re.sub(r'[^a-zA-Z0-9_]', '_', function_name)
        if not function_name[0].isalpha() and function_name[0] != '_':
            function_name = 'cmd_' + function_name

        mcp_script_str = self.llm_provider.generate_mcp_script(
            mcp_name=function_name,
            task_description=task_description,
            args_details="The function should accept no arguments and fulfill the user's request.",
            user_command=user_command
        )

        if not mcp_script_str:
            return None, None

        new_mcp_function, metadata = self.mcp_factory.create_mcp_from_script(function_name, mcp_script_str)

        if new_mcp_function:
            mcp_actual_name = metadata.get("name") or function_name
            self.mcp_box.add_mcp(
                name=mcp_actual_name, 
                function=new_mcp_function,
                description=metadata.get("description", f"Dynamically generated MCP for '{user_command}'."),
                args_info=metadata.get("args", "N/A"),
                returns_info=metadata.get("returns", "N/A"),
                source="dynamically-generated",
                requires=metadata.get("requires"),
                script_content=mcp_script_str,
                original_command=user_command  # Store the original command for reference
            )
            return self.mcp_box.get_mcp(mcp_actual_name), mcp_actual_name
        
        return None, None

    def _attempt_dynamic_mcp_creation_streaming(self, user_command: str):
        """Stream the dynamic MCP creation process for the full user command."""
        # Use the user's original command as the task description
        task_description = f"Handle the user request: '{user_command}'"
        
        # Generate a simple function name based on the command
        command_words = user_command.lower().split()
        if len(command_words) >= 2:
            function_name = "_".join(command_words[:2])
        elif command_words:
            function_name = command_words[0]
        else:
            function_name = "handle_request"
        
        # Clean the function name
        import re
        function_name = re.sub(r'[^a-zA-Z0-9_]', '_', function_name)
        if not function_name[0].isalpha() and function_name[0] != '_':
            function_name = 'cmd_' + function_name

        yield f"🤖 Generating new tool for: '{user_command}'\n"
        yield f"📋 Task: {task_description}\n"
        yield f"🔧 Function name: {function_name}\n\n"
        yield "💭 AI is thinking and writing code:\n"
        yield "```python\n"
        
        # Stream the LLM generation in real-time
        mcp_script_chunks = []
        error_occurred = False
        
        for chunk in self.llm_provider.generate_mcp_script_streaming(
            mcp_name=function_name,
            task_description=task_description,
            args_details="The function should accept no arguments and fulfill the user's request.",
            user_command=user_command
        ):
            if chunk.startswith("Error:"):
                yield f"\n```\n\n❌ {chunk}\n"
                error_occurred = True
                return
            
            yield chunk
            mcp_script_chunks.append(chunk)
        
        yield "\n```\n\n"
        
        if not mcp_script_chunks:
            yield "❌ Failed to generate tool script.\n"
            return

        mcp_script_str = "".join(mcp_script_chunks)
        yield "🔄 Compiling and testing the generated code...\n"
        
        new_mcp_function, metadata = self.mcp_factory.create_mcp_from_script(function_name, mcp_script_str)

        if new_mcp_function:
            mcp_actual_name = metadata.get("name") or function_name
            self.mcp_box.add_mcp(
                name=mcp_actual_name, 
                function=new_mcp_function,
                description=metadata.get("description", f"Dynamically generated tool for '{user_command}'."),
                args_info=metadata.get("args", "N/A"),
                returns_info=metadata.get("returns", "N/A"),
                source="dynamically-generated",
                requires=metadata.get("requires"),
                script_content=mcp_script_str,
                original_command=user_command
            )
            yield f"✅ Successfully created tool '{mcp_actual_name}'!\n"
            yield f"📖 Description: {metadata.get('description', 'No description provided.')}\n"
            yield f"⚙️ Arguments: {metadata.get('args', 'N/A')}\n\n"
            yield f"🚀 Now executing '{mcp_actual_name}'...\n"
            yield "---\n"
            
            # Return the created MCP for execution
            yield ("__MCP_CREATED__", self.mcp_box.get_mcp(mcp_actual_name), mcp_actual_name)
        else:
            yield f"❌ Failed to compile the generated code for '{user_command}'.\n"

    def _extract_intent_and_args(self, command_str: str):
        """Enhanced command parsing that uses LLM to understand natural language queries"""
        command_lower = command_str.lower().strip()
        
        # First, try LLM-based intent parsing for natural language understanding
        try:
            action, args = self.llm_provider.parse_intent(command_str)
            if action:  # If LLM successfully parsed intent
                return action, args
        except Exception as e:
            # If LLM parsing fails, fall back to regex patterns
            pass
        
        # Fallback to regex patterns for common queries (in case LLM is unavailable)
        # IP address patterns
        ip_patterns = [
            r'what.*ip',
            r'my.*ip',
            r'get.*ip',
            r'show.*ip',
            r'ip.*address',
            r'current.*ip'
        ]
        
        # Weather patterns
        weather_patterns = [
            r'\b(weather|temperature|forecast)\b',
            r'what.*weather',
            r'how.*weather',
            r'weather.*like'
        ]
        
        # Time patterns
        time_patterns = [
            r'\b(time|clock)\b',
            r'what.*time',
            r'current.*time',
            r'time.*now'
        ]
        
        # Check for IP-related queries
        for pattern in ip_patterns:
            if re.search(pattern, command_lower):
                return "ip", []
        
        # Check for weather queries        
        for pattern in weather_patterns:
            if re.search(pattern, command_lower):
                return "weather", []
                
        # Check for time queries
        for pattern in time_patterns:
            if re.search(pattern, command_lower):
                return "time", []
        
        # Final fallback to traditional command parsing
        try:
            parts = shlex.split(command_str)
            if not parts:
                return None, []
        except ValueError:
            parts = command_str.split()
            if not parts:
                return None, []
        
        action = parts[0].lower()
        args = parts[1:]
        
        return action, args
    
    def _process_command_internal(self, command_str: str):
        if not command_str.strip():
            return "Please enter a command."

        if command_str.lower().strip() == "help":
            return self.mcp_box.list_mcps()
        
        if command_str.lower().strip() == "quit":
            return "quit_signal"

        # First, try to find an existing MCP that might handle this command
        # Check if any existing MCP's original_command matches or is similar
        for mcp_name, mcp_data in self.mcp_box.mcps.items():
            if mcp_data.get('original_command') == command_str:
                # Exact match found, reuse this MCP
                try:
                    return mcp_data["function"]()
                except Exception as e:
                    return f"Error executing existing tool '{mcp_name}': {e}"

        # No existing MCP found, create a new one
        mcp_data, mcp_name = self._attempt_dynamic_mcp_creation(command_str)
        if not mcp_data:
            return f"Sorry, I couldn't create a new tool to handle '{command_str}' at this time."

        try:
            return mcp_data["function"]()
        except Exception as e:
            return f"Error executing new tool '{mcp_name}': {e}"

    def process_command_streaming(self, command_str: str):
        if not command_str.strip():
            yield "Please enter a command."
            return

        if command_str.lower().strip() == "help":
            yield self.mcp_box.list_mcps()
            return
        
        if command_str.lower().strip() == "quit":
            yield "quit_signal"
            return

        # First, try to find an existing MCP that might handle this command
        for mcp_name, mcp_data in self.mcp_box.mcps.items():
            if mcp_data.get('original_command') == command_str:
                # Exact match found, reuse this MCP
                try:
                    output = mcp_data["function"]()
                    if inspect.isgenerator(output):
                        for chunk in output:
                            yield chunk
                    else:
                        yield output
                    return
                except Exception as e:
                    yield f"Error executing existing tool '{mcp_name}': {e}"
                    return

        # No existing MCP found, create a new one
        mcp_data = None
        mcp_display_name = "new_tool"
        
        for stream_item in self._attempt_dynamic_mcp_creation_streaming(command_str):
            # Check if this is the special signal that MCP was created
            if isinstance(stream_item, tuple) and stream_item[0] == "__MCP_CREATED__":
                mcp_data = stream_item[1]
                mcp_display_name = stream_item[2]
                break
            else:
                # Stream the LLM generation process
                yield stream_item
        
        if not mcp_data:
            yield f"Sorry, I couldn't create a new tool to handle '{command_str}' at this time."
            return

        # Execute the newly created MCP
        try:
            output = mcp_data["function"]()
            if inspect.isgenerator(output):
                for chunk in output:
                    yield chunk
            else:
                yield output
        except Exception as e:
            yield f"Error executing new tool '{mcp_display_name}': {e}"