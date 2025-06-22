#!/usr/bin/env python3
"""
GAIA Benchmark CLI Runner for Open-Alita

Simple command-line interface for running GAIA benchmark tests.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from agent.gaia_agent import GAIAAgent, GAIAQuestion

def main():
    parser = argparse.ArgumentParser(description='Run GAIA benchmark tests with Open-Alita')
    parser.add_argument('jsonl_file', help='Path to JSONL file containing GAIA questions')
    parser.add_argument('--max-questions', type=int, help='Maximum number of questions to process')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    parser.add_argument('--submission', help='Output file for GAIA submission (JSONL format)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.jsonl_file):
        print(f"Error: File {args.jsonl_file} not found")
        sys.exit(1)
    
    print("🚀 Starting GAIA Benchmark with Open-Alita")
    print(f"📁 Questions file: {args.jsonl_file}")
    if args.max_questions:
        print(f"🔢 Max questions: {args.max_questions}")
    print("-" * 50)
    
    # Initialize GAIA agent
    try:
        agent = GAIAAgent()
        print("✅ GAIA Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize GAIA Agent: {e}")
        sys.exit(1)
    
    # Run benchmark
    results = []
    start_time = time.time()
    
    try:
        for result in agent.run_gaia_benchmark(args.jsonl_file, args.max_questions, args.verbose):
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                sys.exit(1)
            
            if "summary" in result:
                # Final summary
                summary = result["summary"]
                print("\n" + "=" * 50)
                print("🏁 BENCHMARK COMPLETE")
                print("=" * 50)
                print(f"📊 Total Questions: {summary['total_questions']}")
                print(f"✅ Correct Answers: {summary['correct_answers']}")
                print(f"🎯 Accuracy: {summary['accuracy']:.2f}%")
                print(f"⏱️  Total Time: {time.time() - start_time:.2f}s")
                print("=" * 50)
                
                # Save results if output file specified
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"💾 Results saved to: {args.output}")
                
                break
            else:
                # Individual question result
                results.append(result)
                
                # Display progress (only if not verbose, since verbose shows real-time)
                if not args.verbose:
                    task_id = result['task_id']
                    is_correct = result['is_correct']
                    expected = result['expected_answer']
                    actual = result['actual_answer']
                    
                    status = "✅" if is_correct else "❌"
                    print(f"{status} {task_id}: Expected '{expected}', Got '{actual}'")
                
                if args.verbose:
                    print(f"   Question: {result['question'][:100]}...")
                    print(f"   Level: {result['level']}")
                    print(f"   Full Response: {result['full_response'][:500]}...")
                    print()
    
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

    # Write submission file if requested
    if args.submission:
        with open(args.submission, 'w') as f:
            for result in results:
                submission_obj = {
                    "task_id": result["task_id"],
                    "model_answer": result["actual_answer"] or ""
                }
                f.write(json.dumps(submission_obj, ensure_ascii=False) + "\n")
        print(f"📝 Submission file written to: {args.submission}")

if __name__ == "__main__":
    main() 