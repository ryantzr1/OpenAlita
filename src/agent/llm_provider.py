import os
import json
from dotenv import load_dotenv
import litellm

class LLMProvider:
    """Placeholder for interacting with an LLM API like DeepSeek."""
    
    def __init__(self, api_key=None, api_url=None):
        load_dotenv()
        # Use OpenAI key + endpoint by default (litellm will route correctly)
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.api_url = api_url if api_url else os.getenv("OPENAI_API_BASE")  # optional

        # Default to an OpenAI chat model; can be changed via env var
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        
        # Store the last generated script for retrieval
        self.last_generated_script = None

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

    def generate_mcp_script_streaming(self, mcp_name, user_command, task_description=None, args_details=None):
        """Generate an MCP script using Claude API with real-time streaming."""
        # Use provided values or derive from user_command
        if task_description is None:
            task_description = f"Handle the user command: {user_command}"
        if args_details is None:
            args_details = "No specific arguments specified"
            
        prompt = self._build_mcp_generation_prompt(
            mcp_name, task_description, args_details, user_command
        )
        
        # Reset and prepare to store the generated script
        script_chunks = []
        first_chunk_processed = False
        try:
            for chunk in self._make_api_call(prompt):
                if not first_chunk_processed:
                    if isinstance(chunk, str) and chunk.startswith("Error:"):
                        yield chunk
                        return
                    first_chunk_processed = True
                script_chunks.append(chunk)
                yield chunk
            
            # Store the complete script for later retrieval
            self.last_generated_script = "".join(script_chunks) if script_chunks else None
            
        except Exception as e:
            yield f"Error: Unexpected error during LLM generation - {str(e)}"
    
    def get_last_generated_mcp_script(self):
        """Retrieve the last generated MCP script."""
        return self.last_generated_script

    def _build_mcp_generation_prompt(self, mcp_name, task_description, args_details, user_command):
        """Build a comprehensive prompt for MCP script generation."""
        
        prompt = f"""You are an expert Python developer tasked with creating a Micro Control Program (MCP) function.

MCP Name to define: {mcp_name}

Task Description: {task_description}

Arguments Details: {args_details}

Original User Command: {user_command}

CRITICAL: You MUST generate a REUSABLE, PARAMETERIZED function that gets the current command dynamically. DO NOT hardcode any command strings.

Please generate a complete, functional Python MCP script that:

1. **Follows this exact format structure:**
   ```
   # MCP Name: {mcp_name}
   # Description: [clear description of what this MCP does]
   # Arguments: [parameters this function can handle]
   # Returns: [what the function returns]
   # Requires: [any imports needed, if applicable]
   
   [import statements if needed]
   
   def {mcp_name}():
       # Get the current command from the global context
       import builtins
       current_command = getattr(builtins, '_current_user_command', '')
       
       # EXTRACT PARAMETERS FROM THE CURRENT COMMAND
       # DO NOT HARDCODE VALUES
       return "result as string"
   ```

2. **CRITICAL Requirements for Reusability:**
   - Get the current command using: `current_command = getattr(builtins, '_current_user_command', '')`
   - Extract parameters from this current_command variable
   - DO NOT hardcode any command strings or specific parameters
   - Make the function work for ANY similar command with different parameters

3. **Parameter Extraction Example for GitHub repositories:**
   ```python
   def github_stars():
       import re
       import requests
       import builtins
       
       try:
           # Get the current command from global context
           current_command = getattr(builtins, '_current_user_command', '')
           if not current_command:
               return "Error: No command provided"
           
           # Extract repository from the current command
           repo_pattern = r'([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)'
           repo_match = re.search(repo_pattern, current_command)
           
           if not repo_match:
               return "Could not find a valid GitHub repository in the command"
           
           owner, repo_name = repo_match.groups()
           repo_full = f"{{owner}}/{{repo_name}}"
           
           # GitHub API request
           api_url = f"https://api.github.com/repos/{{repo_full}}"
           headers = {{'Accept': 'application/vnd.github.v3+json'}}
           response = requests.get(api_url, headers=headers)
           
           if response.status_code == 200:
               repo_data = response.json()
               star_count = repo_data.get('stargazers_count', 0)
               return f"The repository {{repo_full}} has {{star_count}} stars"
           elif response.status_code == 404:
               return f"Repository {{repo_full}} not found on GitHub"
           else:
               return f"GitHub API error: Status code {{response.status_code}}"
               
       except Exception as e:
           return f"Error in github_stars: {{str(e)}}"
   ```

4. **Other Parameter Extraction Examples:**

   For weather queries:
   ```python
   # Get current command and extract location
   current_command = getattr(builtins, '_current_user_command', '')
   words = current_command.lower().split()
   location_words = [w for w in words if w not in ['weather', 'in', 'for', 'at', 'the', 'what', 'is']]
   location = location_words[0] if location_words else "London"
   ```

   For greetings:
   ```python
   # Get current command and extract name
   current_command = getattr(builtins, '_current_user_command', '')
   words = current_command.split()
   name_words = [w for w in words if w.lower() not in ['greet', 'hello', 'hi', 'say']]
   name = name_words[0] if name_words else "friend"
   ```

IMPORTANT: 
- Generate ONLY the complete MCP script
- ALWAYS use `current_command = getattr(builtins, '_current_user_command', '')` to get the current command
- DO NOT hardcode any command strings or parameters
- Make the function truly reusable for different commands with different parameters
"""

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
                max_tokens=3000,
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