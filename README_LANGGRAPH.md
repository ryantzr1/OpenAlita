# Alita Agent System with LangGraph Multi-Agent Workflows

## Overview

Alita is now enhanced with **LangGraph-powered multi-agent workflows** that enable sophisticated task orchestration through specialized agents working collaboratively. The system intelligently routes queries between traditional single-agent processing and advanced multi-agent workflows based on task complexity and requirements.

## 🚀 Key Features

### Multi-Agent Architecture

- **🤖 Orchestrator Agent**: Analyzes tasks, plans workflows, and routes to appropriate agents
- **🔍 Researcher Agent**: Gathers information from web sources and available tools
- **📊 Analyzer Agent**: Processes data, extracts insights, and identifies patterns
- **⚙️ Executor Agent**: Creates and executes specialized tools (MCPs) for specific tasks
- **🎯 Synthesizer Agent**: Combines all information into coherent, comprehensive responses

### Intelligent Routing

The system automatically determines the best approach for each query:

- **Simple queries** → Traditional single-agent processing (fast, efficient)
- **Complex research tasks** → Multi-agent LangGraph workflow (comprehensive, collaborative)
- **Multi-step analysis** → Orchestrated agent collaboration with data flow

### Enhanced Capabilities

- **Graph-based workflow orchestration** with LangGraph
- **Dynamic tool creation** and reuse through MCPs
- **Web search integration** for real-time information
- **Streaming responses** with real-time progress updates
- **Modular, scalable architecture** for easy extension

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key (or compatible LLM provider)
- Required dependencies (see `requirements.txt`)

## 🛠️ Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**

```bash
# Create .env file with your API credentials
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # Optional
LLM_MODEL_NAME=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
```

## 🎮 Usage

### Quick Start Demo

```bash
python demo_langgraph.py
```

This interactive demo showcases:

- Intelligent workflow routing decisions
- Multi-agent collaboration examples
- Performance comparisons between approaches
- Manual query testing capabilities

### Basic Usage

```python
from src.agent.alita_agent import AlitaAgent

# Initialize the enhanced agent
agent = AlitaAgent()

# Simple query (uses traditional agent)
for response in agent.process_command_with_routing("What is 5 + 3?"):
    print(response, end='')

# Complex query (uses multi-agent workflow)
for response in agent.process_command_with_routing(
    "Analyze current AI trends and their impact on software development"
):
    print(response, end='')
```

### Advanced Usage

```python
import asyncio
from src.agent.alita_agent import AlitaAgent

async def advanced_processing():
    agent = AlitaAgent()

    # Check system status
    status = agent.get_workflow_status()
    print(f"LangGraph enabled: {status['langgraph_enabled']}")

    # Explain routing decision
    query = "Research quantum computing applications"
    explanation = agent.explain_workflow_routing(query)
    print(explanation)

    # Process with async support
    result = await agent.process_command_async(query)
    print(result)

asyncio.run(advanced_processing())
```

## 🔄 Workflow Examples

### Simple Query Flow

```
User Query → Routing Decision → Traditional Agent → Response
```

### Multi-Agent Workflow

```
User Query → Orchestrator → Researcher → Analyzer → Synthesizer → Response
                    ↓
                 Executor (if tools needed)
```

### Example Routing Decisions

| Query Type         | Example                                    | Routing                      |
| ------------------ | ------------------------------------------ | ---------------------------- |
| Simple calculation | "What is 15 \* 23?"                        | Traditional Agent            |
| Complex research   | "Analyze AI trends in healthcare"          | Multi-Agent Workflow         |
| Multi-step task    | "Research X, analyze Y, then synthesize Z" | Multi-Agent Workflow         |
| Tool creation      | "Create a currency converter"              | Traditional Agent → Executor |

## 🎯 Agent Specializations

### Orchestrator Agent

- **Purpose**: Task analysis and workflow planning
- **Capabilities**: Complexity assessment, agent routing, workflow orchestration
- **When used**: Entry point for all multi-agent workflows

### Researcher Agent

- **Purpose**: Information gathering and source verification
- **Capabilities**: Web search, tool discovery, data collection
- **When used**: Queries requiring external information or research

### Analyzer Agent

- **Purpose**: Data processing and insight extraction
- **Capabilities**: Pattern recognition, trend analysis, data interpretation
- **When used**: Complex queries with substantial data to process

### Executor Agent

- **Purpose**: Task execution and tool creation
- **Capabilities**: MCP generation, tool execution, specialized computations
- **When used**: Queries requiring specific tools or calculations

### Synthesizer Agent

- **Purpose**: Information integration and response generation
- **Capabilities**: Content synthesis, narrative creation, final formatting
- **When used**: Final step in all multi-agent workflows

## 🔧 Configuration

### Routing Sensitivity

Adjust when queries use multi-agent workflows by modifying `should_use_langgraph()` in `LangGraphWorkflowManager`:

```python
def should_use_langgraph(self, query: str) -> bool:
    # Customize routing logic here
    return indicator_count >= 1 or len(query.split()) > 8
```

### Agent Customization

Each agent can be extended with additional capabilities:

```python
class CustomResearcherAgent(ResearcherAgent):
    def __init__(self, llm_provider, web_agent, mcp_box):
        super().__init__(llm_provider, web_agent, mcp_box)
        self.capabilities.append("custom_capability")

    async def process(self, state: WorkflowState) -> WorkflowState:
        # Custom processing logic
        return await super().process(state)
```

## 📊 Performance Characteristics

### Traditional Agent

- **Best for**: Simple queries, direct commands, tool execution
- **Performance**: Fast response times, minimal overhead
- **Resource usage**: Low memory, single LLM call per task

### Multi-Agent Workflow

- **Best for**: Complex research, multi-step analysis, comprehensive tasks
- **Performance**: Slower but more thorough responses
- **Resource usage**: Higher memory, multiple LLM calls, richer context

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **LangGraph Workflow Failures**

   ```python
   # Check LangGraph initialization
   status = agent.get_workflow_status()
   print(status['langgraph_enabled'])
   ```

3. **API Key Issues**
   ```bash
   # Verify environment variables
   echo $OPENAI_API_KEY
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Adding New Agents

1. Create a new agent class inheriting from `SpecializedAgent`
2. Implement the `process()` method
3. Add the agent to `LangGraphWorkflowManager`
4. Update routing logic as needed

### Extending Capabilities

- Add new tools to the MCP system
- Enhance web search capabilities
- Improve routing intelligence
- Add new workflow patterns

## 📝 License

This project maintains the same license as the original Alita system.

## 🙏 Acknowledgments

- **LangGraph** for the powerful workflow orchestration framework
- **LangChain** for the foundational agent abstractions
- **Original Alita** team for the excellent foundation

---

_The enhanced Alita system with LangGraph multi-agent workflows enables solving real-world problems through intelligent agent collaboration and sophisticated task orchestration._
