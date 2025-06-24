# LangGraph Workflow Package

This package contains the modular components of the Open-Alita LangGraph workflow, organized for better maintainability and separation of concerns.

## Structure

```
workflow/
├── __init__.py          # Package exports and imports
├── state.py             # State definition for the workflow
├── nodes.py             # Core workflow node functions
├── browser_agent.py     # Browser automation node and utilities
└── README.md           # This documentation
```

## Components

### `state.py`

Contains the `State` class that defines the data structure for the LangGraph workflow, including:

- Original query and iteration tracking
- Agent outputs (web results, MCP tools, execution results)
- Evaluation and synthesis data
- Streaming chunks for real-time updates

### `nodes.py`

Contains all the core workflow node functions:

- `coordinator_node`: Analyzes queries and determines workflow strategy
- `web_agent_node`: Performs intelligent web searches with query decomposition
- `mcp_agent_node`: Creates and executes custom tools
- `evaluator_node`: Assesses answer completeness
- `synthesizer_node`: Generates final answers
- `_extract_tool_arguments`: Helper function for tool argument extraction

### `browser_agent.py`

Contains the browser automation functionality:

- `browser_agent_node`: Enhanced browser agent with MCP integration
- `browser_agent_router`: Routes browser results based on success/failure
- Task analysis and description utilities
- Error handling and recovery functions
- Timeout and step calculation helpers

### `__init__.py`

Exports all the necessary components for use in the main workflow file, providing a clean interface for the `LangGraphCoordinator`.

## Benefits of This Structure

1. **Modularity**: Each component has a single responsibility
2. **Maintainability**: Easier to find and modify specific functionality
3. **Testability**: Individual components can be tested in isolation
4. **Readability**: Smaller files are easier to understand
5. **Reusability**: Components can be imported and used elsewhere if needed

## Usage

The main `langgraph_workflow.py` file now imports from this package:

```python
from .workflow import (
    State,
    coordinator_node,
    web_agent_node,
    mcp_agent_node,
    browser_agent_node,
    browser_agent_router,
    evaluator_node,
    synthesizer_node
)
```

This creates a clean separation between the workflow orchestration (in `langgraph_workflow.py`) and the individual workflow components (in this package).
