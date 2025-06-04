# Open-Alita

Open-Alita is a generalist agent designed to enable scalable agentic reasoning with minimal predefinition and maximal self-evolution. This project leverages the Model Context Protocol (MCP) to dynamically create, adapt, and reuse capabilities based on the demands of various tasks, moving away from the reliance on predefined tools and workflows.

## Features

- **Minimal Predefinition**: Equipped with a minimal set of core capabilities, allowing for flexibility and adaptability.
- **Maximal Self-Evolution**: The agent can autonomously create and refine external capabilities as needed.
- **Dynamic MCP Creation**: Alita can generate and adapt MCPs on-the-fly, enhancing its ability to tackle diverse tasks.

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

This project is inspired by the Alita project by CharlesQ9 and the concepts presented in the research paper "Alita: A Large-scale Incremental Task Assigner with Task-Agent Reciprocity and Task-Task Synergy".

- **Original Alita Project:** [CharlesQ9/Alita on GitHub](https://github.com/CharlesQ9/Alita)
- **Research Paper:** [Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution (arXiv:2505.20286)](https://arxiv.org/abs/2505.20286)

Full credits to the authors and contributors of these works for the foundational architecture and ideas.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

We would like to thank the contributors and the community for their support and feedback in developing Open-Alita.
