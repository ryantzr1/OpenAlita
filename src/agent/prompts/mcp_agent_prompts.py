"""
MCP Agent Node Prompts

Prompts used by the MCP agent node for intelligent tool analysis, creation, and execution.
"""

TOOL_REQUIREMENTS_ANALYSIS_PROMPT = """Analyze this user query and determine what tools/functions need to be created to answer it effectively.

USER QUERY: {query}

ANALYSIS TASK:
1. Break down the query into logical components
2. Identify what computational tasks are needed
3. Determine if multiple tools are required
4. Decide the execution order (sequential vs parallel)
5. Identify specialized capabilities needed (APIs, vision, real-time data, etc.)

SPECIALIZED CAPABILITIES TO CONSIDER:
- **Browser Automation**: YouTube videos, dynamic content, authentication
- **Real-time APIs**: GitHub API, YouTube API, Twitter API, weather APIs
- **Vision/Image Processing**: Image analysis, OCR, video processing
- **System Access**: File operations, network info, hardware specs
- **Real-time Data**: Current time, live feeds, dynamic content
- **Authentication**: OAuth flows, API keys, user credentials
- **Interactive Operations**: User input, real-time calculations
- **Platform Integration**: Discord, Slack, email, messaging

RESPONSE FORMAT:
Return a JSON object with this structure:
{{
    "tool_requirements": [
        {{
            "name": "tool_name",
            "description": "What this tool does",
            "purpose": "Why this tool is needed",
            "dependencies": ["list", "of", "other", "tools", "if", "any"],
            "execution_order": 1,
            "can_run_parallel": false,
            "specialized_capabilities": ["api_access", "vision", "real_time", "etc"],
            "api_endpoints": ["specific", "apis", "needed"],
            "external_services": ["github", "youtube", "openai", "etc"]
        }}
    ],
    "execution_strategy": "sequential" | "parallel" | "mixed",
    "estimated_tool_count": 1-5,
    "reasoning": "Explanation of why these tools are needed",
    "requires_external_apis": true/false,
    "requires_vision": true/false,
    "requires_real_time": true/false
}}

Analyze the query: {query}"""

TOOL_SCRIPT_GENERATION_PROMPT = """Create a Python function for this specific tool requirement:

TOOL NAME: {tool_name}
DESCRIPTION: {description}
PURPOSE: {purpose}
QUERY CONTEXT: {query}

SPECIALIZED CAPABILITIES: {specialized_capabilities}
API ENDPOINTS: {api_endpoints}
EXTERNAL SERVICES: {external_services}

IMPORTANT FUNCTION SIGNATURE RULES:
- If the tool needs to process the query text, use: def {tool_name}(text):
- If the tool is self-contained (calculations, counters), use: def {tool_name}():
- If the tool needs multiple parameters, be explicit about what they are

Create a focused, single-purpose function that does exactly what's needed.
Include proper error handling and return meaningful results.

Script format:
# MCP Name: {tool_name}
# Description: {description}
# Arguments: function arguments (text, or none if self-contained)
# Returns: what the function returns
# Requires: comma-separated list of required modules

def {tool_name}():
    # Implementation here with proper error handling
    return result"""

BROWSER_MCP_ANALYSIS_PROMPT = """Analyze this browser automation task and determine if we need to create MCP tools to assist with the browser automation.

BROWSER TASK: {query}

ANALYSIS QUESTIONS:
1. Will this browser task need computational assistance (calculations, data processing, API calls)?
2. Do we need tools to process data extracted from the browser?
3. Are there any calculations or transformations needed on browser data?
4. Do we need tools to interact with external APIs during browser automation?
5. Will we need tools to save, process, or analyze browser results?

EXAMPLES OF MCP TOOLS NEEDED DURING BROWSER AUTOMATION:
- Data processing tools for extracted information
- Calculation tools for financial data, statistics, etc.
- API integration tools for real-time data
- File processing tools for downloads/uploads
- Image analysis tools for screenshots
- Text processing tools for extracted content

Respond with JSON:
{{
    "needs_mcp_tools": true/false,
    "required_tools": [
        {{
            "name": "tool_name",
            "description": "What this tool does during browser automation",
            "purpose": "Why this tool is needed",
            "execution_timing": "before_browser" | "during_browser" | "after_browser"
        }}
    ],
    "reasoning": "Explanation of MCP tool requirements"
}}"""

BROWSER_TOOL_SCRIPT_PROMPT = """Create a Python function that will assist with browser automation:

TOOL NAME: {tool_name}
DESCRIPTION: {description}
PURPOSE: {purpose}
BROWSER TASK: {query}
EXECUTION TIMING: {timing}

This tool should:
- Work seamlessly with browser automation
- Handle data from browser interactions
- Provide computational support for browser tasks
- Include proper error handling

Script format:
# MCP Name: {tool_name}
# Description: {description}
# Arguments: function arguments
# Returns: what the function returns
# Requires: comma-separated list of required modules

def {tool_name}():
    # Implementation here with proper error handling
    return result""" 