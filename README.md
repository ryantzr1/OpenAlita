# Open-Alita

Open-Alita is a generalist agent designed to enable scalable agentic reasoning with minimal predefinition and maximal self-evolution. This project leverages the Model Context Protocol (MCP) to dynamically create, adapt, and reuse capabilities based on the demands of various tasks, moving away from the reliance on predefined tools and workflows.

**Follow updates & connect:** [@ryan_tzr on Twitter](https://x.com/ryan_tzr)

## Inspiration and Credits

This project is inspired by the Alita project by CharlesQ9 and the concepts presented in the research paper "Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution".

- **Original Alita Project:** [CharlesQ9/Alita on GitHub](https://github.com/CharlesQ9/Alita)
- **Research Paper:** [Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution (arXiv:2505.20286)](https://arxiv.org/abs/2505.20286)

Full credits to the authors and contributors of these works for the foundational architecture and ideas.

## 🌟 New: Browser Automation Integration

Open-Alita now integrates with [browser-use](https://github.com/browser-use/browser-use) for powerful browser automation capabilities! The system intelligently routes tasks to the most appropriate agent:

- **🌐 Browser Automation**: YouTube videos, interactive websites, authentication, visual tasks
- **🔍 Web Search**: Simple information queries, static content, documentation
- **🛠️ Tool Creation**: Mathematical calculations, API integrations, data processing
- **🎨 Synthesis**: Combining information into comprehensive answers

## Architecture

Open-Alita implements a modular, self-evolving architecture built around the Model Context Protocol (MCP) and LangGraph workflows. The system is designed to minimize hard-coded behaviors while maximizing adaptive capabilities.

**📊 [View Complete Architecture Diagram](./assets/architecture_diagram.md)**

The architecture consists of multiple layers working together to provide intelligent query routing, dynamic tool creation, web search, and browser automation capabilities.

### Core Components

#### 1. **LangGraph Coordinator** (`src/agent/langgraph_workflow.py`)

The intelligent workflow coordinator that routes tasks to the most appropriate agent.

**Key Features:**

- **Smart Task Routing**: Uses LLM to analyze queries and determine the best approach
- **Browser Automation Detection**: Automatically identifies tasks requiring browser interaction
- **Multi-Agent Orchestration**: Coordinates between web search, tool creation, and browser automation
- **Iterative Refinement**: Continuously improves results through multiple iterations

#### 2. **Browser Agent** (`src/agent/langgraph_workflow.py`)

Handles complex web automation tasks using browser-use.

**Key Features:**

- **Video Content**: YouTube videos, video analysis, watching content
- **Visual Tasks**: Screenshots, image analysis, OCR, visual verification
- **Interactive Websites**: Login forms, shopping carts, social media interactions
- **Dynamic Content**: JavaScript-heavy sites, real-time updates, SPAs
- **Authentication**: OAuth flows, API keys, user credentials
- **Platform Integration**: GitHub, Twitter, LinkedIn, Discord, Slack
- **E-commerce**: Shopping, checkout processes, product browsing
- **Multi-step Workflows**: Job applications, form filling, complex processes

#### 3. **Web Agent** (`src/agent/web_agent.py`)

Handles web search and content extraction for simple information queries.

**Key Features:**

- **Intelligent Query Decomposition**: Breaks complex queries into focused searches
- **Multiple Search Strategies**: Targeted, broader, and verification searches
- **Content Analysis**: Relevance scoring, credibility assessment, summarization
- **Follow-up Searches**: Automatically identifies and fills information gaps

#### 4. **MCP Agent** (`src/agent/langgraph_workflow.py`)

Creates and executes custom tools for computational and data processing tasks.

**Key Features:**

- **Dynamic Tool Creation**: Generates specialized tools based on query requirements
- **Tool Chaining**: Executes multiple tools in sequence or parallel
- **Dependency Management**: Handles tool dependencies and execution order
- **Specialized Capabilities**: API integrations, data analysis, system operations

#### 5. **AlitaAgent** (`src/agent/alita_agent.py`)

The central orchestrator that handles user interactions and coordinates all system components.

**Key Features:**

- **Dynamic Intent Recognition**: Uses LLM-powered natural language understanding
- **Command Processing**: Handles both streaming and batch processing modes
- **MCP Lifecycle Management**: Creates, caches, and reuses dynamic capabilities
- **Fallback Mechanisms**: Gracefully degrades from complex workflows to simple handling

#### 6. **Web Interface** (`src/web/`)

A Flask-based web application providing an intuitive chat interface.

**Components:**

- **WebApp** (`web_app.py`): Flask server with streaming response support
- **Frontend** (`templates/index.html`): Real-time chat interface with streaming
- **Styling** (`static/style.css`): Modern, responsive UI design

### System Flow

```
User Input → LangGraph Coordinator → LLM Analysis
                ↓
    Route to appropriate agent:
    ├── Browser Agent (browser-use) → Complex web automation
    ├── Web Agent → Information search and extraction
    ├── MCP Agent → Tool creation and execution
    └── Synthesizer → Final answer generation
                ↓
    Evaluator → Assess completeness → Iterate if needed
                ↓
    Return comprehensive result
```

## Installation

To set up the project with browser automation support:

```bash
# Clone the repository
git clone https://github.com/ryantzr1/OpenAlita
cd Open-Alita

# Install dependencies
uv sync

# Install browser automation dependencies
uv add browser-use
playwright install chromium --with-deps --no-shell

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# DEEPWISDOM_API_KEY=your_deepwisdom_key (required for all functionality)
# ANTHROPIC_API_KEY=your_anthropic_key (required for browser automation)
# FIRECRAWL_API_KEY=your_firecrawl_key (optional)
```

## Features

- **🌐 Browser Automation**: Full browser control for complex web tasks using browser-use
- **🔍 Intelligent Web Search**: Multi-strategy search with content analysis and follow-ups
- **🛠️ Dynamic Tool Creation**: Generate specialized tools on-demand using MCP
- **🧠 Smart Task Routing**: LLM-powered analysis to choose the best approach
- **🔄 Iterative Refinement**: Continuously improve results through multiple iterations
- **📊 Real-time Streaming**: Watch agents work in real-time with streaming output
- **💾 Capability Reuse**: Automatically cache and reuse previously generated tools
- **🎯 Natural Language Processing**: Advanced intent recognition for intuitive interactions

## Usage

### Web Interface

To run the Alita agent with the web interface:

```bash
cd src/web
python web_app.py
```

This will start the server on `http://127.0.0.1:5001/`.

### Web Search Tasks

```bash
# Information queries
"What is the latest news about AI developments?"
"How many stars does the Python repository have on GitHub?"
"What is the weather forecast for tomorrow?"

# Research
"Find information about the history of machine learning"
"Research the latest trends in web development"
```

### Tool Creation Tasks

```bash
# Calculations
"Calculate the area of a circle with radius 10"
"Solve the quadratic equation x² + 5x + 6 = 0"

# Data processing
"Analyze this dataset and create a summary report"
"Convert this CSV file to JSON format"

# System operations
"Get my current IP address and check if it's in a specific country"
"List all files in the current directory and sort them by size"
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Required for all functionality
DEEPWISDOM_API_KEY=your_deepwisdom_api_key

# Required for browser automation
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional for enhanced web search
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

**Note:** LiteLLM support for multiple LLM providers will be added back in a future update, allowing you to use various AI model providers through a unified interface.

### Browser Setup

Ensure browser automation is properly configured:

```bash
# Install browser-use
uv add browser-use

# Install Playwright browser
playwright install chromium --with-deps --no-shell

# Verify installation
python -c "import browser_use; print('Browser automation ready!')"
```

## Troubleshooting

### Browser Automation Issues

1. **browser-use not installed**:

   ```bash
   uv add browser-use
   ```

2. **Playwright browser not found**:

   ```bash
   playwright install chromium --with-deps --no-shell
   ```

3. **Missing API keys**:

   - Add your API keys to the `.env` file
   - Ensure the keys are valid and have sufficient credits

4. **Browser automation fails**:
   - Check if the task requires authentication
   - Verify the website is accessible
   - Try rephrasing the request

### General Issues

1. **Import errors**: Ensure all dependencies are installed
2. **API rate limits**: Check your API key usage and limits
3. **Memory issues**: Restart the application if it becomes unresponsive

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

We would like to thank the contributors and the community for their support and feedback in developing Open-Alita.

**Special thanks to:**

- **[DeepWisdom](https://www.deepwisdom.ai/)** for providing API credits during the development process - huge huge huge thanks!
- [browser-use](https://github.com/browser-use/browser-use) for powerful browser automation capabilities
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [Model Context Protocol](https://modelcontextprotocol.io/) for dynamic tool creation

## GAIA Testing

Open-Alita includes comprehensive testing capabilities for evaluating agent performance on the GAIA benchmark. The GAIA test suite helps validate the agent's reasoning, tool usage, and problem-solving abilities.

**📋 [GAIA Test Documentation](./GAIA_BENCHMARK_README.md)**

The GAIA tests cover various domains including:

- Mathematical reasoning and calculations
- Web search and information retrieval
- Tool creation and execution
- Multi-step problem solving
- Real-world task completion
