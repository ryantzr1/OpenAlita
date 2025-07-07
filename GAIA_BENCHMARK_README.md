# GAIA Benchmark Testing for Open-Alita

This directory contains the GAIA benchmark testing system for Open-Alita, designed to evaluate the system's performance on the GAIA benchmark questions.

## Overview

The GAIA benchmark tests an AI system's ability to:

- Follow specific output formatting rules
- Provide step-by-step reasoning
- Give precise final answers in a standardized format
- Handle complex multi-step queries requiring web search and tool usage

## Files

- `src/agent/gaia_agent.py` - Core GAIA agent implementation
- `src/gaia_benchmark.py` - CLI runner for benchmark tests
- `test_gaia_sample.jsonl` - Sample questions for testing
- `GAIA_BENCHMARK_README.md` - This file

## Setup Requirements

### Required API Keys

Before running the GAIA benchmark, ensure you have the following API keys set in your environment variables:

```bash
# Required for all functionality
DEEPWISDOM_API_KEY=your_deepwisdom_api_key

# Required for browser automation and LLM access
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional for enhanced web search (recommended for better performance)
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

### Required Files

The GAIA benchmark system requires the following files to be present:

1. **Your GAIA Questions File**: Place your GAIA benchmark questions in JSONL format in the `gaia_files/` directory or specify the full path when running the benchmark.

2. **Sample Test File**: The `test_gaia_sample.jsonl` file is included for testing the system.

3. **Core Implementation Files**: All required implementation files are already present in the `src/` directory.

### File Structure

```
├── gaia_files/                    # Place your GAIA questions here
│   └── your_gaia_questions.jsonl  # Your GAIA benchmark questions
├── src/
│   ├── agent/
│   │   ├── gaia_agent.py         # Core GAIA agent
│   │   └── ...                   # Other agent files
│   └── gaia_benchmark.py         # CLI runner
├── test_gaia_sample.jsonl        # Sample questions for testing
└── GAIA_BENCHMARK_README.md      # This file
```

## Quick Start

### 1. Test with Sample Questions

```bash
# Run the sample questions with recommended flags
python src/gaia_benchmark.py test_gaia_sample.jsonl --verbose --submission sample_submission.jsonl

# Run with limited questions and save results
python src/gaia_benchmark.py test_gaia_sample.jsonl --max-questions 1 --output results.json --submission submission.jsonl --verbose
```

### 2. Run Full Benchmark

```bash
# Run on your GAIA questions file with recommended flags
python src/gaia_benchmark.py test_gaia_sample.jsonl --output benchmark_results.json --submission gaia_submission.jsonl --verbose

python src/gaia_benchmark.py test_gaia_sample.jsonl --output benchmark_results_github.json --submission gaia_submission_github.jsonl --verbose
```

**💡 Recommended Flags:**

- `--verbose` or `-v`: Shows detailed real-time output including step-by-step reasoning
- `--submission`: Creates a GAIA-compatible submission file for official evaluation

## Command Line Options

```bash
python src/gaia_benchmark.py [OPTIONS] jsonl_file

Options:
  jsonl_file              Path to JSONL file containing GAIA questions
  --max-questions INT     Maximum number of questions to process
  --output PATH           Output file for results (JSON format)
  --submission PATH       Output file for GAIA submission (JSONL format)
  --verbose, -v           Verbose output showing full questions
  --help                  Show help message
```

**⭐ Recommended Flags:**

- `--verbose` or `-v`: Provides detailed real-time output including step-by-step reasoning, making it easier to debug and understand the system's decision-making process
- `--submission`: Creates a GAIA-compatible submission file that can be used for official GAIA benchmark evaluation

## GAIA Question Format

Each line in the JSONL file should contain a JSON object with:

```json
{
  "task_id": "unique-identifier",
  "Question": "The actual question text",
  "Level": 1,
  "Final answer": "expected_answer",
  "file_name": "",
  "Annotator Metadata": {
    "Steps": "Step-by-step solution...",
    "Number of steps": "10",
    "How long did this take?": "5 minutes",
    "Tools": "1. Web browser\n2. Search engine",
    "Number of tools": "2"
  }
}
```

## Output Format

The system outputs results in real-time and saves a JSON file with:

```json
[
  {
    "task_id": "unique-identifier",
    "question": "The question text",
    "expected_answer": "expected_answer",
    "actual_answer": "system_answer",
    "is_correct": true,
    "level": 1,
    "full_response": "Complete system response with reasoning"
  }
]
```

## System Features

### GAIA-Compliant Output

- Follows exact GAIA system prompt format
- Outputs "FINAL ANSWER:" template
- Handles numbers, strings, and comma-separated lists correctly

### Multi-Agent Workflow

- Uses LangGraph coordinator for complex reasoning
- Web search capabilities via Firecrawl
- Dynamic tool creation via MCP
- Evaluation and synthesis agents

### Streaming Output

- Real-time progress updates
- Step-by-step reasoning display
- Error handling and recovery

## Example Output

```
🚀 Starting GAIA Benchmark with Open-Alita
📁 Questions file: test_gaia_sample.jsonl
🔢 Max questions: 2
--------------------------------------------------
✅ GAIA Agent initialized successfully

Task ID: c61d22de-5f6c-4958-a7f6-5e9707bd3466
Question: A paper about AI regulation that was originally submitted to arXiv.org in June 2022...
Level: 2

Let me think through this step by step:

🚀 Starting Multi-Agent Workflow
📝 Query: A paper about AI regulation that was originally submitted to arXiv.org in June 2022...

🤖 Coordinator starting...
🔍 Found 5 web results
✅ Coordinator completed (2.3s)

🤖 Web Agent starting...
⚡ Tool Output: Search results for AI regulation arXiv June 2022
✅ Web Agent completed (1.8s)

🤖 Synthesizer starting...
Based on the search results, I found the paper "Fairness in Agreement With European Values: An Interdisciplinary Perspective on AI Regulation" which contains a figure with three axes. The labels are: deontological, egalitarian, localized, standardized, utilitarian, and consequential.

Now I need to search for Physics and Society articles from August 11, 2016 to find which of these words appears...

✅ Synthesizer completed (3.2s)

FINAL ANSWER: egalitarian

✅ c61d22de-5f6c-4958-a7f6-5e9707bd3466: Expected 'egalitarian', Got 'egalitarian'

==================================================
🏁 BENCHMARK COMPLETE
==================================================
📊 Total Questions: 2
✅ Correct Answers: 2
🎯 Accuracy: 100.00%
⏱️  Total Time: 45.23s
==================================================
```

## Requirements

- Python 3.8+
- All Open-Alita dependencies
- Firecrawl API key for web search functionality
- OpenAI/Anthropic API key for LLM access

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **API Key Issues**: Ensure your API keys are set in environment variables
3. **File Not Found**: Check that the JSONL file path is correct
4. **Memory Issues**: Use `--max-questions` to limit processing for large files

### Debug Mode

For detailed debugging, you can modify the logging level in `src/agent/gaia_agent.py`:

```python
logger = logging.getLogger('alita.gaia')
logger.setLevel(logging.DEBUG)
```

## Performance Tips

1. **Use Recommended Flags**: Always use `--verbose` and `--submission` for better debugging and official evaluation
2. **Batch Processing**: Use `--max-questions` to test with subsets first
3. **Result Saving**: Always use `--output` to save results for analysis
4. **Verbose Mode**: Use `--verbose` for detailed question analysis and debugging
5. **Submission File**: Use `--submission` to create GAIA-compatible output for official evaluation
6. **Error Recovery**: The system continues processing even if individual questions fail

## Integration with Full GAIA Benchmark

To integrate with the official GAIA benchmark:

1. Download the official GAIA questions JSONL file
2. Run the benchmark: `python src/gaia_benchmark.py gaia_questions.jsonl --output results.json`
3. Analyze results using the JSON output
4. Compare with official GAIA evaluation metrics

The system is designed to be compatible with the official GAIA evaluation framework and output format.
