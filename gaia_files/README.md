# GAIA Files Directory

This directory is where you should place your GAIA benchmark questions files.

## What to Add Here

Place your GAIA benchmark questions in JSONL format in this directory. Each line should contain a JSON object with the following structure:

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

## Example Usage

```bash
# Place your questions file here
cp your_gaia_questions.jsonl gaia_files/

# Run the benchmark
python src/gaia_benchmark.py gaia_files/your_gaia_questions.jsonl --output results.json
```

## Getting GAIA Questions

You can obtain the official GAIA benchmark questions from:

- The official GAIA benchmark repository
- Academic papers and research publications
- Contact the GAIA benchmark maintainers

## Testing

For testing purposes, you can use the included sample file:

```bash
python src/gaia_benchmark.py test_gaia_sample.jsonl --verbose
```
