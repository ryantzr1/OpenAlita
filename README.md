# Open-Alita

Open-Alita is a generalist agent designed to enable scalable agentic reasoning with minimal predefinition and maximal self-evolution. This project leverages the Model Context Protocol (MCP) to dynamically create, adapt, and reuse capabilities based on the demands of various tasks, moving away from the reliance on predefined tools and workflows.

**Follow updates & connect:** [@ryan_tzr on Twitter](https://x.com/ryan_tzr)

## Architecture

Open-Alita implements a modular, self-evolving architecture built around the Model Context Protocol (MCP). The system is designed to minimize hard-coded behaviors while maximizing adaptive capabilities.

**📊 [View Complete Architecture Diagram](./assets/architecture_diagram.md)**

The architecture consists of multiple layers working together to provide intelligent query routing, dynamic tool creation, and web search capabilities.

### Core Components

#### 1. **AlitaAgent** (`src/agent/alita_agent.py`)

The central orchestrator that handles user interactions and coordinates all system components.

**Key Features:**

- **Dynamic Intent Recognition**: Uses LLM-powered natural language understanding combined with regex fallbacks
- **Command Processing**: Handles both streaming and batch processing modes
- **MCP Lifecycle Management**: Creates, caches, and reuses dynamic capabilities
- **Fallback Mechanisms**: Gracefully degrades from LLM-based parsing to pattern matching

#### 2. **MCPBox** (`src/agent/mcp_box.py`)

A capability registry that stores and manages all Model Context Protocol functions.

**Key Features:**

- **Capability Storage**: Maintains both pre-loaded and dynamically generated MCPs
- **Metadata Management**: Tracks function descriptions, arguments, return types, and source origins
- **Command Caching**: Stores original user commands to enable capability reuse
- **Introspection**: Provides detailed listings of available capabilities

#### 3. **MCPFactory** (`src/agent/mcp_factory.py`)

A secure code execution engine that transforms LLM-generated scripts into executable functions.

**Key Features:**

- **Script Parsing**: Extracts metadata and code from LLM responses
- **Sandboxed Execution**: Provides restricted Python execution environment
- **Dependency Management**: Safely imports whitelisted modules based on requirements
- **Error Handling**: Graceful failure and recovery mechanisms

#### 4. **LLMProvider** (`src/agent/llm_provider.py`)

Handles all interactions with large language models for code generation and intent parsing.

**Key Features:**

- **Code Generation**: Creates new MCP scripts based on user requests
- **Streaming Support**: Real-time code generation with user feedback
- **Intent Parsing**: Natural language understanding for command interpretation

#### 5. **Web Interface** (`src/web/`)

A Flask-based web application providing an intuitive chat interface.

**Components:**

- **WebApp** (`web_app.py`): Flask server with streaming response support
- **Frontend** (`templates/index.html`): Real-time chat interface with streaming
- **Styling** (`static/style.css`): Modern, responsive UI design

### System Flow

```
User Input → AlitaAgent → Intent Recognition
                ↓
    Check MCPBox for existing capability
                ↓
    If not found: LLMProvider → Generate Code → MCPFactory → Validate & Execute
                ↓
    Store in MCPBox for future reuse
                ↓
    Execute capability → Return result
```

### Security Model

- **Sandboxed Execution**: All dynamically generated code runs in a restricted Python environment
- **Whitelisted Imports**: Only approved modules can be imported based on declared requirements
- **Safe Globals**: Limited built-in functions available to generated code
- **Error Isolation**: Failures in generated code don't crash the main system

## Features

- **Minimal Predefinition**: Equipped with a minimal set of core capabilities, allowing for flexibility and adaptability.
- **Maximal Self-Evolution**: The agent can autonomously create and refine external capabilities as needed.
- **Dynamic MCP Creation**: Alita can generate and adapt MCPs on-the-fly, enhancing its ability to tackle diverse tasks.
- **Real-time Streaming**: Watch capabilities being created and executed in real-time
- **Capability Reuse**: Automatically caches and reuses previously generated tools
- **Natural Language Processing**: Advanced intent recognition for intuitive interactions

## Upcoming Features

### Core Improvements

- **Enhanced Error Handling** - Better robustness for dynamic code generation
- **Performance Optimization** - Code caching and faster execution
- **Persistent Memory** - Save learned capabilities across sessions
- **Multi-modal Support** - Handle images, audio, and files
- **Web Agent** - Browse websites, extract content, and interact with web services (coming this week)

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
cd Open-Alita
pip install -r requirements.txt
```

## Usage

To run the Alita agent with the web interface:

1.  Navigate to the web application directory:
    ```bash
    cd src/web
    ```
2.  Run the Flask web application:
    ```bash
    python web_app.py
    ```

This will typically start the server on `http://127.0.0.1:5001/`.

To run the agent in command-line mode (without the web UI):

```bash
python -m src.prompt.mcp
```

Make sure to configure the necessary parameters (e.g., API keys in a `.env` file or directly in the code) as needed.

## Inspiration and Credits

This project is inspired by the Alita project by CharlesQ9 and the concepts presented in the research paper "Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution".

- **Original Alita Project:** [CharlesQ9/Alita on GitHub](https://github.com/CharlesQ9/Alita)
- **Research Paper:** [Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution (arXiv:2505.20286)](https://arxiv.org/abs/2505.20286)

Full credits to the authors and contributors of these works for the foundational architecture and ideas.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

We would like to thank the contributors and the community for their support and feedback in developing Open-Alita.
