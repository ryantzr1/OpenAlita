import os
import json
from dotenv import load_dotenv
import litellm  # Unified LLM client supporting OpenAI models via the same interface

class LLMProvider:
    """Placeholder for interacting with an LLM API like DeepSeek."""
    
    def __init__(self, api_key=None, api_url=None):
        load_dotenv()
        # Use OpenAI key + endpoint by default (litellm will route correctly)
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.api_url = api_url if api_url else os.getenv("OPENAI_API_BASE")  # optional

        # Default to an OpenAI chat model; can be changed via env var
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

    def generate_mcp_script(self, mcp_name, task_description, args_details, user_command):
        """Generate an MCP script using Claude API based on the provided parameters."""
        prompt = self._build_mcp_generation_prompt(
            mcp_name, task_description, args_details, user_command
        )
        
        script_chunks = []
        first_chunk_processed = False
        try:
            for chunk in self._make_api_call(prompt):
                if not first_chunk_processed:
                    if isinstance(chunk, str) and chunk.startswith("Error:"):
                        return None
                    first_chunk_processed = True
                script_chunks.append(chunk)
            
            if not script_chunks:
                return None

            return "".join(script_chunks)
        except Exception as e:
            return None

    def generate_mcp_script_streaming(self, mcp_name, task_description, args_details, user_command):
        """Generate an MCP script using Claude API with real-time streaming."""
        prompt = self._build_mcp_generation_prompt(
            mcp_name, task_description, args_details, user_command
        )
        
        first_chunk_processed = False
        try:
            for chunk in self._make_api_call(prompt):
                if not first_chunk_processed:
                    if isinstance(chunk, str) and chunk.startswith("Error:"):
                        yield chunk
                        return
                    first_chunk_processed = True
                yield chunk
        except Exception as e:
            yield f"Error: Unexpected error during LLM generation - {str(e)}"


    def _build_mcp_generation_prompt(self, mcp_name, task_description, args_details, user_command):
        """Build a comprehensive prompt for MCP script generation."""
        
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
   ```

Generate ONLY the MCP script with proper formatting. Do not include any explanatory text before or after the script."""

        return prompt
    
    def _make_api_call(self, prompt_text):
        """Stream completion tokens using LiteLLM (defaults to OpenAI provider)."""

        try:
            response_stream = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                api_key=self.api_key,
                api_base=self.api_url,
                temperature=0.5,
                max_tokens=1500,
                stream=True,
            )

            for chunk in response_stream:
                # LiteLLM returns either object or dict depending on version/provider
                delta_text = None
                try:
                    delta_text = chunk.choices[0].delta.content  # type: ignore[attr-defined]
                except AttributeError:
                    try:
                        delta_text = chunk["choices"][0].get("delta", {}).get("content")
                    except (KeyError, TypeError):
                        delta_text = None

                if delta_text:
                    yield delta_text

        except Exception as e:
            yield f"Error: LLM request failed - {str(e)}"

    def parse_intent(self, user_command):
        """Use the LLM to parse user intent and extract action/args from natural language."""
        prompt = f"""You are an intent parser for a command-line assistant. Parse the user's natural language command and extract:
1. The main action/intent (single word, lowercase)
2. Any relevant arguments

User command: "{user_command}"

Respond ONLY with a JSON object in this exact format:
{{"action": "intent_name", "args": ["arg1", "arg2"]}}


"""

        try:
            # Get the response from the LLM
            response_chunks = []
            for chunk in self._make_api_call(prompt):
                if isinstance(chunk, str) and chunk.startswith("Error:"):
                    return None, []
                response_chunks.append(chunk)
            
            if not response_chunks:
                return None, []
            
            response_text = "".join(response_chunks).strip()
            
            # Parse the JSON response
            import json
            try:
                parsed = json.loads(response_text)
                action = parsed.get("action", "").lower()
                args = parsed.get("args", [])
                return action, args
            except json.JSONDecodeError:
                # Fallback: try to extract action from partial response
                if "action" in response_text and ":" in response_text:
                    # Simple fallback parsing
                    lines = response_text.split('\n')
                    for line in lines:
                        if '"action"' in line and ':' in line:
                            action_part = line.split(':')[1].strip(' ",')
                            return action_part.lower(), []
                return None, []
                
        except Exception as e:
            return None, []