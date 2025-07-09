"""
GAIA Benchmark Agent for Open-Alita

This module provides a specialized agent for GAIA benchmark testing with:
- GAIA-compliant system prompt
- JSONL question processing
- Structured output format with "FINAL ANSWER:" template
- Multi-step reasoning capabilities
- File handling for attached documents
"""

import json
import logging
import time
import os
import sys
import pandas as pd
from typing import Dict, Any, List, Generator, Optional
from dataclasses import dataclass
import uuid

# Add project root to sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .llm_provider import LLMProvider
from .web_agent import WebAgent
from .mcp_factory import MCPFactory
from .langgraph_workflow import LangGraphCoordinator

logger = logging.getLogger('alita.gaia')

# GAIA System Prompt (exact format from benchmark)
GAIA_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to put in the list is a number or a string."""

@dataclass
class GAIAQuestion:
    """Structure for GAIA benchmark questions"""
    task_id: str
    question: str
    level: int
    final_answer: str
    file_name: str
    annotator_metadata: Dict[str, Any]

class GAIAAgent:
    """Specialized agent for GAIA benchmark testing"""
    
    def __init__(self):
        logger.info("Initializing GAIA Agent")
        self.llm_provider = LLMProvider()
        self.web_agent = WebAgent()
        self.mcp_factory = MCPFactory()
        self.langgraph_coordinator = LangGraphCoordinator()
        # self.gaia_files_dir = "gaia_files"
        self.gaia_files_dir = "/root/OpenAlita/gaia_dataset/2023/validation"
        logger.info("GAIA Agent initialized successfully")
    
    def _load_file_content(self, file_name: str) -> Optional[str]:
        """Load and process file content from gaia_files directory"""
        if not file_name:
            return None
            
        file_path = os.path.join(self.gaia_files_dir, file_name)
        
        # Check if it's a URL (starts with http)
        if file_name.startswith('http'):
            return self._download_and_process_file(file_name)
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            # Handle different file types
            if file_name.lower().endswith('.xlsx') or file_name.lower().endswith('.xls'):
                return self._process_excel_file(file_path)
            elif file_name.lower().endswith('.csv'):
                return self._process_csv_file(file_path)
            elif file_name.lower().endswith('.txt'):
                return self._process_text_file(file_path)
            elif file_name.lower().endswith('.pdb'):
                return self._process_pdb_file(file_path)
            elif file_name.lower().endswith('.pdf'):
                return self._process_pdf_file(file_path)
            elif file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                return self._process_image_file(file_path)
            elif file_name.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac')):
                # Transcribe audio using OpenAI Whisper API
                from src.utils import transcribe_audio_openai
                transcript = transcribe_audio_openai(file_path)
                if transcript:
                    return f"[AUDIO TRANSCRIPT]\n{transcript}"
                else:
                    return None
            else:
                logger.warning(f"Unsupported file type: {file_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
    
    def _download_and_process_file(self, url: str) -> Optional[str]:
        """Download file from URL and process it"""
        try:
            # Check if it's a Hugging Face URL
            if "huggingface.co" in url and "/datasets/gaia-benchmark/GAIA/" in url:
                return self._download_from_huggingface(url)
            else:
                # Fallback to requests for other URLs
                return self._download_with_requests(url)
                
        except Exception as e:
            logger.error(f"Error downloading file from {url}: {e}")
            return None
    
    def _download_from_huggingface(self, url: str) -> Optional[str]:
        """Download file from Hugging Face GAIA dataset"""
        try:
            from huggingface_hub import hf_hub_download
            import tempfile
            import os
            
            # Extract filename from URL
            # URL format: https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/filename.pdb
            filename = url.split('/')[-1]
            file_path_in_repo = '/'.join(url.split('/')[-3:-1]) + '/' + filename  # e.g., "2023/validation/filename.pdb"
            
            logger.info(f"Downloading from Hugging Face: {filename}")
            
            # Download using huggingface_hub
            local_path = hf_hub_download(
                repo_id="gaia-benchmark/GAIA",
                filename=file_path_in_repo,
                repo_type="dataset"
            )
            
            # Process the downloaded file
            if filename.lower().endswith('.pdb'):
                content = self._process_pdb_file(local_path)
            elif filename.lower().endswith('.txt'):
                content = self._process_text_file(local_path)
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                content = self._process_image_file(local_path)
            else:
                content = self._process_text_file(local_path)  # Default to text
            
            return content
            
        except Exception as e:
            logger.error(f"Error downloading from Hugging Face: {e}")
            return None
    
    def _download_with_requests(self, url: str) -> Optional[str]:
        """Download file using requests (fallback method)"""
        try:
            import requests
            import tempfile
            import os
            
            logger.info(f"Downloading file from: {url}")
            
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get filename from URL
            filename = url.split('/')[-1]
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            
            # Process based on file extension
            if filename.lower().endswith('.pdb'):
                content = self._process_pdb_file(tmp_file_path)
            elif filename.lower().endswith('.txt'):
                content = self._process_text_file(tmp_file_path)
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                content = self._process_image_file(tmp_file_path)
            else:
                content = self._process_text_file(tmp_file_path)  # Default to text
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return content
            
        except Exception as e:
            logger.error(f"Error downloading file from {url}: {e}")
            return None
    
    def _process_excel_file(self, file_path: str) -> str:
        """Process Excel file and return content as string"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            content_parts = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                content_parts.append(f"Sheet: {sheet_name}")
                content_parts.append(df.to_string(index=False))
                content_parts.append("")  # Empty line between sheets
            
            return "\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {e}")
            return f"Error reading Excel file: {str(e)}"
    
    def _process_csv_file(self, file_path: str) -> str:
        """Process CSV file and return content as string"""
        try:
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return f"Error reading CSV file: {str(e)}"
    
    def _process_text_file(self, file_path: str) -> str:
        """Process text file and return content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return f"Error reading text file: {str(e)}"
    
    def _process_pdb_file(self, file_path: str) -> str:
        """Process PDB file and return content as string"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add a header to identify it as PDB content
            return f"PDB File Content:\n{content}"
            
        except Exception as e:
            logger.error(f"Error reading PDB file {file_path}: {e}")
            return f"Error reading PDB file: {str(e)}"
    
    def _process_image_file(self, file_path: str) -> str:
        """Process image file - provide basic image information for MCP vision analysis"""
        try:
            from PIL import Image
            
            # Get basic image information
            image = Image.open(file_path)
            width, height = image.size
            format_type = image.format
            mode = image.mode
            
            image_info = f"""Image file: {os.path.basename(file_path)}
Image details:
- Format: {format_type}
- Dimensions: {width}x{height} pixels
- Color mode: {mode}
- File size: {os.path.getsize(file_path)} bytes
- Full path: {file_path}

This is an image file that may contain visual information relevant to the question.
The MCP agent can create a vision analysis tool to extract text or analyze the image content if needed."""
            
            return image_info
                
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {e}")
            return f"Image file: {os.path.basename(file_path)}\nError processing image: {str(e)}"
    
    def _create_file_context_prompt(self, question: str, file_content: str) -> str:
        """Create a prompt that includes file content for the agent"""
        
        # Check if this is an image file
        if "Image file:" in file_content and "MCP agent can create a vision analysis tool" in file_content:
            return f"""You have access to an image file that contains relevant information for answering the question.

Question: {question}

Image File Information:
{file_content}

IMPORTANT: This is an image file. If the question requires analyzing the visual content of this image, you should:
1. Use the MCP agent to create a vision analysis tool
2. The tool can use GPT-4 Vision or similar to analyze the image
3. Extract text, identify objects, or answer questions about the visual content

Please analyze the question and determine if you need to create a vision analysis tool to process this image."""
        else:
            return f"""You have access to a file that contains relevant information for answering the question.

Question: {question}

File Content:
{file_content}

IMPORTANT: When extracting information from the file content, please preserve the exact names, descriptions, and terminology as they appear in the source. Do not simplify or abbreviate terms unless specifically requested. Pay attention to descriptive adjectives and qualifiers.

Please analyze the file content and answer the question based on the information provided in the file, maintaining the exact terminology used."""
    
    def process_gaia_question(self, question: GAIAQuestion) -> Generator[str, None, None]:
        """Process a GAIA question with streaming output"""
        logger.info(f"Processing GAIA question: {question.task_id}")
        
        # Start with reasoning
        yield f"Task ID: {question.task_id}\n"
        yield f"Question: {question.question}\n"
        yield f"Level: {question.level}\n"
        
        # Handle file attachment
        specific_image_file = None
        enhanced_question = question.question  # Default to original question
        
        if question.file_name:
            yield f"File: {question.file_name}\n"
            
            # Check if it's an image file
            if question.file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                # For image files, get the full path to pass to LangGraph
                image_path = os.path.join(self.gaia_files_dir, question.file_name)
                if os.path.exists(image_path):
                    specific_image_file = image_path
                    yield f"📁 Image file found: {question.file_name}\n"
                else:
                    yield f"⚠️ Warning: Image file not found: {image_path}\n"
                    specific_image_file = None
            else:
                # For non-image files, load content as before
                file_content = self._load_file_content(question.file_name)
                if file_content:
                    yield f"📁 File loaded successfully ({len(file_content)} characters)\n"
                    # Create enhanced prompt with file content
                    enhanced_question = self._create_file_context_prompt(question.question, file_content)
                else:
                    yield f"⚠️ Warning: Could not load file {question.file_name}\n"
                    enhanced_question = question.question
        
        yield "\nLet me think through this step by step:\n\n"
        
        try:
            # Use LangGraph workflow for comprehensive analysis
            # Pass the specific image file if available
            full_response = ""
            if specific_image_file:
                # Pass the specific image file to LangGraph
                for chunk in self.langgraph_coordinator.process_query_streaming(enhanced_question, specific_image_file):
                    full_response += chunk
                    yield chunk
            else:
                # No specific image, use normal processing
                for chunk in self.langgraph_coordinator.process_query_streaming(enhanced_question):
                    full_response += chunk
                    yield chunk
            
            # Extract final answer using LLM with GAIA format
            final_answer = self._extract_gaia_final_answer(question.question, full_response)
            
            # Output in GAIA format
            yield f"\n\nFINAL ANSWER: {final_answer}\n"
            
            logger.info(f"GAIA question {question.task_id} completed with answer: {final_answer}")
            
        except Exception as e:
            error_msg = f"Error processing GAIA question: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield f"\nError: {error_msg}\n"
            yield "FINAL ANSWER: ERROR\n"
    
    def _extract_gaia_final_answer(self, question: str, full_response: str) -> str:
        """Extract final answer in GAIA format using LLM"""
        prompt = f"""Based on the question and the comprehensive analysis below, provide the FINAL ANSWER in the exact GAIA format.

Question: {question}

Analysis: {full_response}

Rules for FINAL ANSWER:
1. Should be a number OR as few words as possible OR a comma separated list
2. If it's a number, don't use comma or units ($, %, etc.) unless specified
3. If it's a string, don't use articles or abbreviations, write digits in plain text
4. If it's a comma separated list, apply the above rules for each element

Provide ONLY the final answer (no explanation, no quotes, just the answer):"""

        try:
            response_chunks = []
            for chunk in self.llm_provider._make_api_call(prompt):
                if isinstance(chunk, str) and chunk.startswith("Error:"):
                    logger.warning(f"LLM error extracting final answer: {chunk}")
                    break
                response_chunks.append(chunk)
            
            if response_chunks:
                final_answer = "".join(response_chunks).strip()
                # Clean up the response
                final_answer = final_answer.replace('"', '').replace("'", "").strip()
                return final_answer
            else:
                return "ERROR"
                
        except Exception as e:
            logger.error(f"Error extracting final answer: {e}")
            return "ERROR"
    
    def load_gaia_questions(self, jsonl_file_path: str) -> List[GAIAQuestion]:
        """Load GAIA questions from JSONL file"""
        questions = []
        try:
            with open(jsonl_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        question = GAIAQuestion(
                            task_id=data.get('task_id', f'unknown_{line_num}'),
                            question=data.get('Question', ''),
                            level=data.get('Level', 1),
                            final_answer=data.get('Final answer', ''),
                            file_name=data.get('file_name', ''),
                            annotator_metadata=data.get('Annotator Metadata', {})
                        )
                        questions.append(question)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
            
            logger.info(f"Loaded {len(questions)} GAIA questions from {jsonl_file_path}")
            return questions
            
        except Exception as e:
            logger.error(f"Error loading GAIA questions from {jsonl_file_path}: {e}")
            return []
    
    def run_gaia_benchmark(self, jsonl_file_path: str, max_questions: Optional[int] = None, verbose: bool = False, existing_tasks: Optional[set] = None, resume: bool = False, correct_answers: int=0) -> Generator[Dict[str, Any], None, None]:
        """Run GAIA benchmark on questions from JSONL file"""
        logger.info(f"Starting GAIA benchmark with file: {jsonl_file_path}")
        
        questions = self.load_gaia_questions(jsonl_file_path)
        if not questions:
            yield {"error": "No questions loaded from file"}
            return
        
        if max_questions:
            questions = questions[:max_questions]
        
        total_questions = len(questions)
        skipped_count = 0  # 添加跳过计数
        
        for i, question in enumerate(questions, 1):
            # 在处理问题前检查是否需要跳过
            if resume and existing_tasks and question.task_id in existing_tasks:
                skipped_count += 1
                if verbose:
                    logger.info(f"Skipping question {i}/{total_questions}: {question.task_id} (already answered)")
                continue
                
            logger.info(f"Processing question {i}/{total_questions}: {question.task_id}")
            
            # Process the question
            response_chunks = []
            if verbose:
                print(f"\n{'='*60}")
                print(f"Processing Question {i}/{total_questions}")
                print(f"{'='*60}")
            
            for chunk in self.process_gaia_question(question):
                response_chunks.append(chunk)
                if verbose:
                    print(chunk, end='', flush=True)
            
            full_response = "".join(response_chunks)
            
            # Extract the final answer from response
            final_answer_match = None
            if "FINAL ANSWER:" in full_response:
                final_answer_part = full_response.split("FINAL ANSWER:")[-1].strip()
                final_answer_match = final_answer_part.split('\n')[0].strip()
            
            # Check if answer is correct
            is_correct = False
            if final_answer_match:
                # Normalize answers for comparison
                expected = str(question.final_answer).strip().lower()
                actual = str(final_answer_match).strip().lower()
                is_correct = expected == actual
                
                if is_correct:
                    correct_answers += 1
            
            result = {
                "task_id": question.task_id,
                "question": question.question,
                "expected_answer": question.final_answer,
                "actual_answer": final_answer_match,
                "is_correct": is_correct,
                "level": question.level,
                "full_response": full_response
            }
            
            # Always yield the result for submission file generation
            yield result
            
            # Progress update
            accuracy = (correct_answers / (i )) * 100 if (i) > 0 else 0
            logger.info(f"Progress: {i}/{total_questions}, Accuracy: {accuracy:.2f}%")
        
        # Final summary
        final_accuracy = (correct_answers / total_questions) * 100
        summary = {
            "summary": {
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "accuracy": final_accuracy,
                "skipped_questions": skipped_count,  
                "benchmark_complete": True
            }
        }
        
        logger.info(f"GAIA benchmark completed. Final accuracy: {final_accuracy:.2f}%")
        yield summary 