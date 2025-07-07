#!/usr/bin/env python3
"""
GAIA Benchmark CLI Runner for Open-Alita

Simple command-line interface for running GAIA benchmark tests.
"""

import sys
import os
os.environ["BROWSER_USE_CHROMIUM_SANDBOX"] = "false"
import argparse
import json
import time
from pathlib import Path

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from agent.gaia_agent import GAIAAgent, GAIAQuestion

def load_existing_submission(submission_file: str) -> set:
    """Load existing task IDs from submission file to enable resuming"""
    existing_tasks = set()
    if os.path.exists(submission_file):
        try:
            with open(submission_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        existing_tasks.add(data.get('task_id', ''))
            print(f"📋 Found {len(existing_tasks)} existing answers in submission file")
        except Exception as e:
            print(f"⚠️ Warning: Could not read existing submission file: {e}")
    return existing_tasks

def write_submission_entry(submission_file: str, task_id: str, model_answer: str):
    """Write a single submission entry to the file"""
    try:
        submission_obj = {
            "task_id": task_id,
            "model_answer": model_answer or ""
        }
        with open(submission_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(submission_obj, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"❌ Error writing to submission file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run GAIA benchmark tests with Open-Alita')
    parser.add_argument('jsonl_file', help='Path to JSONL file containing GAIA questions')
    parser.add_argument('--max-questions', type=int, help='Maximum number of questions to process')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    parser.add_argument('--submission', help='Output file for GAIA submission (JSONL format)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--resume', action='store_true', help='Resume from existing submission file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.jsonl_file):
        print(f"Error: File {args.jsonl_file} not found")
        sys.exit(1)
    
    print("🚀 Starting GAIA Benchmark with Open-Alita")
    print(f"📁 Questions file: {args.jsonl_file}")
    if args.max_questions:
        print(f"🔢 Max questions: {args.max_questions}")
    if args.submission:
        print(f"📝 Submission file: {args.submission}")
        if args.resume:
            print("🔄 Resume mode enabled")
    print("-" * 50)
    
    # Load existing submission if resuming
    existing_tasks = set()
    if args.submission and args.resume:
        existing_tasks = load_existing_submission(args.submission)
    
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
    processed_count = 0
    skipped_count = 0
    correct_answers = 0  # Track correct answers for real-time accuracy
    
    try:
        for result in agent.run_gaia_benchmark(args.jsonl_file, args.max_questions, args.verbose, skip_tasks=existing_tasks):
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
                if args.resume:
                    print(f"⏭️  Skipped (already answered): {skipped_count}")
                print("=" * 50)
                
                # Save results if output file specified
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"💾 Results saved to: {args.output}")
                
                break
            else:
                # Individual question result
                task_id = result['task_id']
                print(f"\n Task ID: {task_id}")
                
                # Skip if already processed and resuming
                if result.get('skipped', False):
                    skipped_count += 1
                    continue
                
                # Write to submission file immediately if specified
                if args.submission:
                    success = write_submission_entry(args.submission, task_id, result['actual_answer'])
                    if success:
                        print(f"💾 Saved answer for {task_id}")
                    else:
                        print(f"❌ Failed to save answer for {task_id}")
                
                results.append(result)
                processed_count += 1
                
                # Update accuracy tracking
                if result['is_correct']:
                    correct_answers += 1
                
                # Calculate current accuracy
                current_accuracy = (correct_answers / processed_count) * 100 if processed_count > 0 else 0
                
                # Display progress with accuracy
                if not args.verbose:
                    is_correct = result['is_correct']
                    expected = result['expected_answer']
                    actual = result['actual_answer']
                    
                    status = "✅" if is_correct else "❌"
                    print(f"{status} {task_id}: Expected '{expected}', Got '{actual}' | Accuracy: {current_accuracy:.1f}% ({correct_answers}/{processed_count})")
                else:
                    # In verbose mode, show accuracy after each question
                    is_correct = result['is_correct']
                    status = "✅" if is_correct else "❌"
                    print(f"\n{status} Question {processed_count} completed | Accuracy: {current_accuracy:.1f}% ({correct_answers}/{processed_count})")
                    print(f"   Question: {result['question'][:100]}...")
                    print(f"   Level: {result['level']}")
                    print(f"   Expected Answer: {result['expected_answer']}")
                    print(f"   Actual Answer: {result['actual_answer']}")
                    # print(f"   Full Response: {result['full_response'][:500]}...")
                    print()
    
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        print(f"📊 Progress saved: {processed_count} questions processed")
        if args.submission:
            print(f"📝 Partial submission file saved to: {args.submission}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"📊 Progress saved: {processed_count} questions processed")
        if args.submission:
            print(f"📝 Partial submission file saved to: {args.submission}")
        sys.exit(1)

    # Final global stats (for all processed)
    total_correct = sum(1 for r in results if r.get("is_correct"))
    total_questions = len(results)
    global_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0

    print("\n" + "=" * 50)
    print(f"🌐 Global Stats (for all processed):")
    print(f"   Total Questions: {total_questions}")
    print(f"   Total Correct Answers: {total_correct}")
    print(f"   Global Accuracy: {global_accuracy:.2f}%")
    print("=" * 50)

    # Note: No need to write submission file at the end since we're writing incrementally
    if args.submission:
        print(f"📝 All answers saved incrementally to: {args.submission}")

if __name__ == "__main__":
    main() 