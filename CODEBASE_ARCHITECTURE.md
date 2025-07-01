# Open-Alita Codebase Architecture

## Overview

Open-Alita is an intelligent multi-agent system that combines web search, browser automation, and dynamic tool creation to answer user queries. The system uses a LangGraph workflow to coordinate between different specialized agents, each handling specific types of tasks.

## Core Architecture

### AlitaAgent (`src/agent/alita_agent.py`)

The main orchestrator that:

- Initializes all components (LLM provider, MCP factory, web agent, LangGraph coordinator)
- Handles basic arithmetic commands directly
- Delegates complex queries to the LangGraph workflow
- Provides fallback mechanisms if the workflow fails

## LangGraph Workflow System

### Coordinator (`src/agent/langgraph_workflow.py` + `src/agent/workflow/nodes.py`)

The intelligent router that analyzes queries and determines the best strategy:

**Key Features:**

- Uses LLM to analyze query intent and requirements
- Determines whether to use web search, browser automation, or tool creation
- Tracks iteration count and information quality
- Handles image files for vision-enabled queries
- Provides fallback heuristics if LLM analysis fails

**Decision Logic:**

- **Browser Automation**: For interactive tasks, video content, login scenarios
- **Web Search**: For current events, factual questions, news
- **Tool Creation**: For computational tasks, system operations
- **Synthesis**: When sufficient information is gathered

### Web Agent (`src/agent/web_agent.py`)

Handles web search and content extraction:

**Capabilities:**

- Uses Firecrawl API for enhanced search with content scraping
- Intelligent query decomposition and follow-up searches
- Content analysis and credibility scoring
- Relevance scoring and result enhancement

**Key Methods:**

- `can_handle_with_search()`: Determines if query is suitable for web search
- `search_web()`: Performs enhanced search with content analysis
- `answer_query()`: Synthesizes search results into coherent answers

### Browser Agent (`src/agent/workflow/browser_agent.py`)

Handles browser automation using browser-use:

**Capabilities:**

- Uses browser-use library for web automation
- Supports vision-enabled tasks with GPT-4
- Handles interactive web tasks (login, clicking, form filling)
- Captures browser automation logs and results
- Provides fallback to web search on failures

**Configuration:**

- Implements timeout and step limits
- Captures execution logs for debugging

### MCP Tools System (`src/agent/mcp_factory.py`)

Creates dynamic tools from LLM-generated code:

**Current Implementation:**

- Generates Python functions from LLM prompts
- Uses regex-based code cleaning
- Direct `exec()` execution of generated code
- Automatic package installation via uv/pip
- Multi-stage code repair and validation

## GAIA Benchmark Agent System

### GAIA Agent (`src/agent/gaia_agent.py`)

Specialized agent for GAIA benchmark testing with comprehensive file handling:

**Key Features:**

- **GAIA-Compliant System Prompt**: Uses exact GAIA benchmark format
- **Multi-Format File Support**: Handles Excel, CSV, images, audio
- **Structured Output**: Enforces "FINAL ANSWER:" template format
- **File Context Integration**: Automatically loads and processes attached files
- **Hugging Face Integration**: Downloads files from GAIA dataset (needs Hugging Face login)

**File Processing Capabilities:**

```python
# Supported file types and processing methods
- Excel files (.xlsx, .xls) → pandas processing
- CSV files → pandas DataFrame analysis
- Image files → vision-enabled analysis
- Audio files → OpenAI Whisper transcription
- Text files → direct content processing
```

**Workflow Steps:**

1. **Question Loading** (`load_gaia_questions()`):

   - Reads JSONL file with GAIA questions
   - Parses task_id, question, level, final_answer, file_name
   - Validates question structure and metadata

2. **File Context Processing** (`_load_file_content()`):

   - Checks for attached files in `gaia_files` directory
   - Downloads files from Hugging Face GAIA dataset if needed
   - Processes files based on their type and format
   - Integrates file content into question context

3. **Enhanced Question Processing** (`process_gaia_question()`):

   - Creates file-aware prompts with context
   - Uses GAIA system prompt for consistent formatting
   - Handles multi-step reasoning for complex questions
   - Extracts final answers using regex pattern matching

4. **Answer Extraction** (`_extract_gaia_final_answer()`):
   - Searches for "FINAL ANSWER:" pattern in response
   - Validates answer format and completeness
   - Handles edge cases and malformed responses
   - Returns clean, benchmark-compatible answers

**Example Workflow:**

```python
# 1. Load GAIA question with attached PDB file
question = GAIAQuestion(
    task_id="2023_validation_001",
    question="What is the resolution of this protein structure?",
    file_name="protein_structure.pdb"
)

# 2. Process attached file
file_content = agent._load_file_content("protein_structure.pdb")
# Downloads from Hugging Face and extracts PDB data

# 3. Create context-aware prompt
context_prompt = agent._create_file_context_prompt(
    question.question,
    file_content
)

# 4. Generate GAIA-compliant response
response = agent.llm_provider._make_api_call(context_prompt)
# Returns: "The resolution is 2.1 Å. FINAL ANSWER: 2.1"

# 5. Extract final answer
final_answer = agent._extract_gaia_final_answer(question.question, response)
# Returns: "2.1"
```

**Benchmark Execution** (`run_gaia_benchmark()`):

- Processes questions in batches with progress tracking
- Supports resume functionality for interrupted runs
- Provides real-time accuracy metrics
- Generates submission files in GAIA format
- Handles errors gracefully with detailed logging

### GAIA Benchmark Runner (`src/gaia_benchmark.py`)

Command-line interface for running GAIA benchmark tests:

**Features:**

- **Resume Support**: Continues from existing submission files
- **Progress Tracking**: Real-time accuracy and completion metrics
- **Flexible Output**: JSON results and JSONL submission formats
- **Verbose Mode**: Detailed logging for debugging
- **Error Handling**: Graceful handling of failures and interruptions

**Usage Example:**

```bash
# Run GAIA benchmark with resume support
python src/gaia_benchmark.py test_gaia_sample.jsonl \
    --max-questions 10 \
    --submission my_submission.jsonl \
    --resume \
    --verbose
```

**Output Format:**

```json
{
  "task_id": "2023_validation_001",
  "model_answer": "2.1",
  "is_correct": true,
  "expected_answer": "2.1",
  "level": 1,
  "question": "What is the resolution of this protein structure?",
  "full_response": "Based on the PDB file... FINAL ANSWER: 2.1"
}
```

## Identified Weak Points in MCP Tools System

### 🎯 Problem Statement

The current MCP tool system has several critical limitations that make it error-prone and difficult to scale:

1. **Regex-based Code Cleaning**: Limited and fragile approach to code validation
2. **Direct Execution**: Uses `exec()` without proper sandboxing or validation
3. **Poor Error Handling**: Minimal validation of code quality or parameter correctness
4. **Scalability Issues**: Difficult to maintain and extend the current approach

### Current Issues

#### 1. Code Generation and Validation

```python
# Current approach in mcp_factory.py
def _clean_script(self, script_content: str) -> Optional[str]:
    # Uses regex patterns for cleaning
    # Limited validation capabilities
    # No file-based validation
```

#### 2. Direct Execution

```python
# Current execution method
exec(cleaned_script, safe_globals)  # Direct exec without proper validation
```

#### 3. Error-Prone Architecture

- Little validation of code quality
- No parameter type checking
- Minimal security considerations
- Difficult debugging and maintenance

#### 4. Not compatible with official MCP

**Current Schema Format** (what we have now):

```json
[
  {
    "name": "calculate_area",
    "description": "Calculate area of a rectangle",
    "metadata": {
      "name": "calculate_area",
      "description": "Calculate area of a rectangle",
      "args": "length (float), width (float)",
      "returns": "area (float)",
      "requires": "math"
    },
    "script_content": "# MCP Name: calculate_area...",
    "created_at": "2025-01-21T16:51:33.394638",
    "usage_count": 2,
    "last_used": "2025-01-21T16:51:33.450005"
  }
]
```

**Problem**: This format is not compatible with the official MCP protocol standards.

## 💡 Proposed Solution: Two-Phase Architecture

### Phase 1: Code Generation and Validation

**Goal**: Replace regex-based code cleaning with proper file-based generation and validation

**Key Changes**:

- Generate tool code to separate files instead of in-memory strings
- Use AST-based validation instead of regex patterns
- Implement proper security checks and code quality analysis
- Create a more robust and maintainable code generation system

### Phase 2: Parameter Generation and Execution

**Goal**: Replace direct `exec()` with schema-based parameter validation and safe execution

**Key Changes**:

- Generate parameters using proper JSON schemas
- Validate parameters against tool requirements before execution
- Implement safe execution environment with proper sandboxing
- Make the system compatible with official MCP protocol standards

## 🏗️ Proposed Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Tool      │───▶│  Code Generator  │───▶│  File Writer    │
│   Request       │    │  & Validator     │    │  (Python/JS)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Parameter     │◀───│  Schema-Based    │◀───│  Agent Call     │
│   Execution     │    │  Parameter Gen   │    │  for Params     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Two Phase Execution Model

**Phase 1**: Tool Creation and Validation

- Generate tool code to files with proper validation
- Ensure code quality and security standards
- Create MCP-compatible tool schemas

**Phase 2**: Parameter Generation and Safe Execution

- Generate parameters using JSON schemas
- Validate parameters before execution
- Execute tools in safe environment
- Pass results to next workflow step

### Proposed MCP Protocol JSON Format

**Target Schema** (what we want to implement):

```json
{
  "tools": [
    {
      "name": "tool_name",
      "description": "Tool description",
      "inputSchema": {
        "type": "object",
        "properties": {
          "parameter_name": {
            "type": "string",
            "description": "Parameter description"
          }
        },
        "required": ["parameter_name"]
      }
    }
  ]
}
```

**Benefits**:

- Compatible with official MCP protocol standards
- Better parameter validation and type checking
- Supports complex data types and nested objects
- Enables proper tool discovery and integration

### Parameter Schema

**Enhanced parameter handling** will support:

- Type validation (string, number, boolean, array, object)
- Required vs optional parameters
- Parameter descriptions and examples
- Nested object validation
- Array validation with item types

## 🔍 Questions for Further Investigation

### PDF Upload Support

**Question**: Does the DeepWisdom API support PDF upload for document analysis? This would be valuable for the GAIA agent when processing PDF files that contain complex documents, charts, or multi-page content that requires advanced document understanding capabilities.
