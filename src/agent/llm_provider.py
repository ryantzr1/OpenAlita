import os
import json
import logging
import requests
import base64
from dotenv import load_dotenv

# Configure logging: write to both console and log file
logging.basicConfig(
    level=logging.INFO,  # 可选调节: DEBUG / INFO / WARNING / ERROR / CRITICAL
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("llm_provider.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class LLMProvider:
    """Placeholder for interacting with an LLM API like DeepSeek."""
    
    def __init__(self, api_key=None, api_url=None):
        load_dotenv()
        # Default to use DeepWisdom API endpoint with Claude
        self.llm_api_key = api_key if api_key else os.getenv("LLM_API_KEY")
        self.api_url = api_url if api_url else os.getenv("LLM_API_BASE", "https://oneapi.deepwisdom.ai/v1")

        # Default to Claude 3.5 Sonnet; can be changed via env var
        self.model_name = os.getenv("LLM_MODEL_NAME", "claude-3-5-sonnet-20241022")
        
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
        
        prompt = f"""You are an expert Python developer tasked with creating a Module Context Protocol (MCP) function.

MCP Name to define: {mcp_name}

Task Description: {task_description}

Arguments Details: {args_details}

Original User Command: {user_command}

CRITICAL: You MUST generate a REUSABLE, PARAMETERIZED function that accepts the query as a parameter. DO NOT rely on global variables.

IMPORTANT FUNCTION NAMING RULES:
- Use ONLY ASCII characters (a-z, A-Z, 0-9, underscore)
- NO Unicode characters, special symbols, or punctuation
- NO spaces, hyphens, or special characters in function names
- Use snake_case naming convention
- Keep names descriptive but concise

VISION TASK GUIDELINES:
- **DO NOT** import or use openai, PIL, or other external vision libraries
- **DO NOT** try to access image files directly or make API calls
- **DO** return a description of what the tool would analyze or extract
- **DO** mention that the system's built-in vision capabilities will handle the actual image processing
- The system already has vision capabilities built into LLMProvider that can analyze images automatically

Please generate a complete, functional Python MCP script that:

1. **Follows this exact format structure:**
   ```
   # MCP Name: {mcp_name}
   # Description: [clear description of what this MCP does]
   # Arguments: [parameters this function can handle]
   # Returns: [what the function returns]
   # Requires: [any imports needed, if applicable]
   
   [import statements if needed]
   
   def {mcp_name}(query=""):
       # EXTRACT PARAMETERS FROM THE QUERY PARAMETER
       # DO NOT HARDCODE VALUES
       # DO NOT RELY ON GLOBAL VARIABLES
       return "result as string"
   ```

2. **CRITICAL Requirements for Reusability:**
   - Accept the query as a parameter: `def {mcp_name}(query=""):`
   - Extract parameters from this query parameter
   - DO NOT hardcode any command strings or specific parameters
   - Make the function work for ANY similar command with different parameters
   - DO NOT use global variables or builtins._current_user_command

3. **Parameter Extraction Example for GitHub repositories:**
   ```python
   def github_stars(query=""):
       import re
       import requests
       
       try:
           if not query:
               return "Error: No query provided"
           
           # Extract repository from the query parameter
           repo_pattern = r'([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)'
           repo_match = re.search(repo_pattern, query)
           
           if not repo_match:
               return "Could not find a valid GitHub repository in the query"
           
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
   # Get query and extract location
   words = query.lower().split()
   location_words = [w for w in words if w not in ['weather', 'in', 'for', 'at', 'the', 'what', 'is']]
   location = location_words[0] if location_words else "London"
   ```

   For greetings:
   ```python
   # Get query and extract name
   words = query.split()
   name_words = [w for w in words if w.lower() not in ['greet', 'hello', 'hi', 'say']]
   name = name_words[0] if name_words else "friend"
   ```

   For calculations:
   ```python
   # Extract numbers from query
   import re
   numbers = re.findall(r'\d+(?:\.\d+)?', query)
   if len(numbers) >= 2:
       length = float(numbers[0])
       width = float(numbers[1])
   ```

CRITICAL FUNCTION NAMING REQUIREMENTS:
- Use ONLY: a-z, A-Z, 0-9, underscore (_)
- NO: spaces, hyphens, apostrophes, quotes, or special characters
- NO: Unicode characters like curly quotes (', ') or dashes (–, —)
- Examples of GOOD names: calculate_area, get_weather, analyze_data
- Examples of BAD names: calculate-area, get'weather, analyze–data

IMPORTANT: 
- Generate ONLY the complete MCP script
- ALWAYS use `def {mcp_name}(query=""):` as the function signature
- DO NOT hardcode any command strings or parameters
- DO NOT use global variables or builtins._current_user_command
- Make the function truly reusable for different commands with different parameters
- Use ONLY ASCII characters in function names
- For vision tasks: Return descriptions, not implementation details
"""

        return prompt
    
    def _make_api_call(self, prompt_text, image_files=None):
        """Make direct API call to DeepWisdom Claude endpoint with streaming, supporting both text and vision."""
        
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        # Build content array for messages
        content = [{"type": "text", "text": prompt_text}]
        
        # Add images if provided
        if image_files:
            logging.info(f"Processing {len(image_files)} image files for vision analysis")
            for image_path in image_files:
                if os.path.exists(image_path):
                    try:
                        logging.info(f"Loading image: {image_path}")
                        with open(image_path, "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            
                        # Determine image type from file extension
                        file_ext = os.path.splitext(image_path)[1].lower()
                        if file_ext == '.png':
                            image_type = "image/png"
                        elif file_ext in ['.jpg', '.jpeg']:
                            image_type = "image/jpeg"
                        elif file_ext == '.gif':
                            image_type = "image/gif"
                        elif file_ext == '.webp':
                            image_type = "image/webp"
                        else:
                            image_type = "image/png"  # Default
                        
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_type};base64,{image_data}"
                            }
                        })
                        logging.info(f"Successfully added image {image_path} to content")
                    except Exception as e:
                        logging.error(f"Error processing image {image_path}: {e}")
                else:
                    logging.error(f"Image file not found: {image_path}")
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 9000,
            "stream": True
        }
        
        logging.info(f"Making API call with {len(content)} content items (text + {len(image_files) if image_files else 0} images)")
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = f"Error: API request failed with status {response.status_code}: {response.text}"
                logging.error(error_msg)
                yield error_msg
                return
            
            # Parse streaming response
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data_part = line_text[6:]  # Remove 'data: ' prefix
                        
                        if data_part.strip() == '[DONE]':
                            break
                            
                        try:
                            chunk_data = json.loads(data_part)
                            choices = chunk_data.get('choices', [])
                            
                            if choices and len(choices) > 0:
                                delta = choices[0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    yield content
                                    
                        except json.JSONDecodeError:
                            # Skip malformed JSON chunks
                            continue
                            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: Network request failed - {str(e)}"
            logging.error(error_msg)
            yield error_msg
        except Exception as e:
            error_msg = f"Error: API call failed - {str(e)}"
            logging.error(error_msg)
            yield error_msg

    def _make_vision_api_call(self, prompt_text, image_files):
        """Make vision-enabled API call with image files."""
        return self._make_api_call(prompt_text, image_files)

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

    def simple_completion(self, prompt, image_files=None):
        """
        Get a single string completion from the LLM (non-streaming).
        """
        response_chunks = []
        for chunk in self._make_api_call(prompt, image_files):
            if isinstance(chunk, str) and chunk.startswith("Error:"):
                raise RuntimeError(chunk)
            response_chunks.append(chunk)
        return "".join(response_chunks).strip()