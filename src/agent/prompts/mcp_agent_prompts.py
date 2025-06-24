"""
MCP Agent Node Prompts

Simplified prompts for creating focused, single-purpose tools.
"""

TOOL_REQUIREMENTS_ANALYSIS_PROMPT = """Analyze this query and determine what tools are needed to solve it.

USER QUERY: {query}

RULES:
1. **FOCUSED TOOLS** - Create only the tools needed to solve the problem
2. **NO REDUNDANCY** - Don't create multiple tools that do similar things
3. **SIMPLE LOGIC** - Use basic Python operations, not complex algorithms
4. **CLEAR PURPOSE** - Each tool should have one clear job

EXAMPLES OF GOOD TOOLS:
- `count_even_numbers` - counts even numbers in a list
- `extract_text_from_image` - extracts text from image (uses built-in vision)
- `calculate_percentage` - calculates percentage from two numbers
- `parse_addresses` - extracts and processes address data

EXAMPLES OF BAD TOOLS:
- `address_parser` + `parity_checker` + `sunset_awning_counter` (redundant!)
- `complex_data_analyzer` (too vague)
- `multi_purpose_processor` (does too many things)

RESPONSE FORMAT:
{{
    "tool_requirements": [
        {{
            "name": "tool_name",
            "description": "What this tool does",
            "purpose": "Why this tool is needed",
            "execution_order": 1,
            "can_run_parallel": false
        }}
    ],
    "execution_strategy": "sequential",
    "reasoning": "Why these tools are sufficient"
}}

Analyze: {query}"""

TOOL_SCRIPT_GENERATION_PROMPT = """Create a simple Python function for this specific task:

TOOL NAME: {tool_name}
DESCRIPTION: {description}
PURPOSE: {purpose}
USER QUERY: {query}

Create a focused, single-purpose function:

```python
# MCP Name: {tool_name}
# Description: {description}
# Arguments: query (string) - the user query to process
# Returns: processed result
# Requires: re, json, math (or other built-in modules only)

def {tool_name}(query=""):
    try:
        # Simple, focused processing logic
        # Extract relevant information from query
        # Perform the specific calculation/processing
        # Return clear result
        
        return result
    except Exception as e:
        return f"Error: {{str(e)}}"
```

Focus on simplicity and clarity. The function should do ONE thing well."""

BROWSER_MCP_ANALYSIS_PROMPT = """Analyze if this browser task needs MCP tools for data processing.

BROWSER TASK: {query}

SIMPLE RULE: Only create tools if you need to process/calculate data from browser results.

EXAMPLES:
- Browser extracts prices → Tool calculates total
- Browser gets text → Tool extracts specific info
- Browser downloads file → Tool processes file content

Respond with JSON:
{{
    "needs_mcp_tools": true/false,
    "required_tools": [
        {{
            "name": "tool_name",
            "description": "What this tool does",
            "purpose": "Why needed",
            "execution_timing": "before_browser" | "during_browser" | "after_browser"
        }}
    ],
    "reasoning": "Simple explanation"
}}"""

BROWSER_TOOL_SCRIPT_PROMPT = """Create a simple Python function for browser data processing:

TOOL NAME: {tool_name}
DESCRIPTION: {description}
PURPOSE: {purpose}
BROWSER TASK: {query}
EXECUTION TIMING: {timing}

Create a simple function that processes data from browser automation:

```python
# MCP Name: {tool_name}
# Description: {description}
# Arguments: query (string) - browser data to process
# Returns: processed result
# Requires: re (or other built-in modules)

def {tool_name}(query=""):
    try:
        # Simple processing logic
        return result
    except Exception as e:
        return f"Error: {{str(e)}}"
```""" 