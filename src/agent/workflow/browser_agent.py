"""
Browser Agent Node

Contains the browser agent node and related browser automation functionality.
"""

import json
import logging
import time
import os
from typing import Dict, Any, List, Literal
from langgraph.types import Command
from dotenv import load_dotenv
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use import Agent

from .state import State
from .context_utils import limit_browser_results

logger = logging.getLogger('alita.langgraph')


def browser_agent_router(state: State) -> Literal["evaluator", "web_agent"]:
    """Route browser agent results based on success/failure"""
    
    # 潜在问题说明 (Potential Issues):
    # 1. 关键词匹配可能过于简单 - 任何包含"error"、"failed"、"loop"的结果都会触发回退
    #    (Keyword matching might be too simple - any result containing "error", "failed", "loop" triggers fallback)
    # 2. 可能产生误报 - 例如"feedback loop"、"loop through results"等正常情况也会被误判
    #    (May cause false positives - e.g., "feedback loop", "loop through results" will be misclassified)
    # 3. 忽略了部分成功的情况 - 即使浏览器自动化部分成功，任何错误都会导致回退到网络搜索
    #    (Ignores partial success - any error causes fallback even if browser automation was partially successful)
    # 4. 没有考虑错误的严重程度 - 轻微错误和严重错误被同等对待
    #    (Doesn't consider error severity - minor and critical errors are treated equally)
    # 5. 可能错过有用的结果 - 即使有最终结果，任何错误也会触发回退
    #    (May miss useful results - any error triggers fallback even if there's a final result)
    
    browser_results = state.get("browser_results", [])
    
    for result in browser_results:
        if isinstance(result, str) and any(keyword in result.lower() for keyword in ["Browser automation failed", "infinite loop", "loop", "unknown", "failed", "error"]):
            return "web_agent"  # Fallback to web search only on actual failures
    
    return "evaluator"  # Normal flow - let evaluator decide

def browser_agent_node(state: State) -> Command[Literal["evaluator", "web_agent"]]:
    """Simplified browser agent node that captures browser-use's native logging"""
    logger.info("Browser agent starting...")

    # Load environment variables
    load_dotenv()
    query = state["original_query"]
     
    chunks = [f"🌐 **Browser Agent:** Starting browser automation\n"]
    chunks.append(f"📝 **Task:** {query}\n")
    
    try:
        # Check if browser-use is available
        try:
            import browser_use
        except ImportError:
            chunks.append("❌ **Error:** browser-use not installed\n")
            chunks.append("💡 **Install with:** pip install browser-use\n")
            
            return Command(
                update={
                    "browser_results": ["Browser automation failed: browser-use not installed"],
                    "streaming_chunks": chunks
                },
                goto="evaluator"
            )
        
        openai_api_key = os.getenv("LLM_API_KEY", "")
        
        if not openai_api_key:
            chunks.append("❌ **Browser automation requires LLM API key**\n")
            chunks.append("💡 **Add to .env file:** LLM_API_KEY=your_openai_api_key\n")
            
            return Command(
                update={
                    "browser_results": ["Browser automation failed: OpenAI API key required"],
                    "streaming_chunks": chunks
                },
                goto="evaluator"
            )
        
        chunks.append("🚀 **Starting browser automation...**\n")
        
        # Simple task description (like main branch)
        task_description = f"""
        Solve this TASK: {query}
        """
        
        try:
            # Use GPT-4 with optimized settings for browser automation
            chat_model = ChatOpenAI(
                model="gpt-4o",
                base_url="https://oneapi.deepwisdom.ai/v1",
                api_key=openai_api_key,
                temperature=0,
            )
                        
            chunks.append("🤖 **Using GPT-4**\n")

            extend_system_message = """
            SKIP YOUTUBE ADS INSTRUCTIONS:
            - To skip a Youtube ad, click the Skip button
            - The Skip button is the button that says "Skip" and is usually located at the bottom of the video player
            - You can also click the Skip button to skip the ad
            """
            
            
            # Create browser-use agent
            agent = Agent(
                task=task_description,
                llm=chat_model,
                use_vision=True,
                extend_system_message=extend_system_message,
                available_file_paths=["downloads"]
            )
            
            chunks.append("✅ **Browser agent configured**\n")
            
            # Execute browser automation
            import asyncio
            import time
            import sys
            import io
            
            # Enhanced timeout calculation - longer timeout for complex tasks
            timeout_seconds = int(os.getenv("BROWSER_TIMEOUT_SECONDS", "500"))  # Configurable timeout
            max_steps = int(os.getenv("BROWSER_MAX_STEPS", "60"))              # Configurable max steps
            
            chunks.append(f"🎯 **Executing browser automation...**\n")
            chunks.append(f"⏱️ **Timeout:** {timeout_seconds}s, **Max Steps:** {max_steps}\n")
            chunks.append("📺 **Browser automation output:**\n")
            chunks.append("=" * 50 + "\n")
            
            # Capture browser-use output
            start_time = time.time()
            
            # Create a custom logger handler to capture browser-use output
            import logging
            from io import StringIO
            
            # Create string buffer to capture logs
            log_capture_string = StringIO()
            ch = logging.StreamHandler(log_capture_string)
            ch.setLevel(logging.INFO)
            
            # Add handler to browser-use logger
            browser_logger = logging.getLogger('browser_use')
            browser_logger.addHandler(ch)
            browser_logger.setLevel(logging.INFO)
            
            try:
                # Run browser automation with enhanced timeout handling
                chunks.append("🔄 **Starting browser automation...**\n")
                
                result = asyncio.run(asyncio.wait_for(
                    agent.run(max_steps=max_steps),
                    timeout=timeout_seconds
                ))
                
                execution_time = time.time() - start_time
                
                # Get captured browser-use logs
                log_contents = log_capture_string.getvalue()
                if log_contents:
                    # Add browser-use logs to chunks
                    for log_line in log_contents.split('\n'):
                        if log_line.strip():
                            chunks.append(f"🤖 {log_line.strip()}\n")
                
                chunks.append("=" * 50 + "\n")
                chunks.append("✅ **Browser automation completed successfully!**\n")
                chunks.append(f"⏱️ **Execution Time:** {execution_time:.1f}s\n")
                
                # Extract simple results with size limits
                browser_results = []
                
                try:
                    # Get final result using proper API
                    final_result = result.final_result()
                    if final_result:
                        # Limit size to prevent payload issues
                        result_text = str(final_result)[:2000]
                        browser_results.append(f"Final result: {result_text}")
                    
                    # Check for errors
                    if result.has_errors():
                        errors = result.errors()
                        if errors:
                            browser_results.append(f"Errors encountered: {len(errors)} errors")
                    
                    # Get visited URLs
                    urls = result.urls()
                    if urls:
                        browser_results.append(f"URLs visited: {len(urls)}")
                        if len(urls) <= 3:
                            browser_results.append(f"Sites: {', '.join(urls)}")
                    
                    # Get action count
                    actions = result.action_names()
                    if actions:
                        browser_results.append(f"Actions executed: {len(actions)}")
                        
                except Exception as api_error:
                    logger.warning(f"Error accessing browser-use result API: {api_error}")
                    browser_results.append(f"Result extraction completed with API access issues")
                
                # Apply browser results limit to prevent context overflow
                browser_results = limit_browser_results(browser_results, max_results=5)
                
                # Ensure total results don't get too large
                if len(str(browser_results)) > 10000:  # 10KB limit
                    browser_results = browser_results[:5]  # Keep only first 2 results
                    browser_results.append("Result: Browser automation completed (truncated for size)")
                
                return Command(
                    update={
                        "browser_results": browser_results,
                        "streaming_chunks": chunks
                    },
                    goto="evaluator"
                )
                
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                chunks.append("=" * 50 + "\n")
                chunks.append(f"⏰ **Browser automation timed out after {timeout_seconds}s**\n")
                chunks.append(f"⏱️ **Execution Time:** {execution_time:.1f}s\n")
                chunks.append("💡 **Note:** Task may have been partially completed\n")
                
                # Try to extract partial results even on timeout
                browser_results = []
                try:
                    # Check if we have any partial results from browser-use
                    if 'result' in locals():
                        final_result = result.final_result()
                        if final_result:
                            result_text = str(final_result)[:2000]
                            browser_results.append(f"Partial result (timeout): {result_text}")
                        
                        urls = result.urls()
                        if urls:
                            browser_results.append(f"URLs visited (partial): {len(urls)}")
                        
                        actions = result.action_names()
                        if actions:
                            browser_results.append(f"Actions executed (partial): {len(actions)}")
                except:
                    pass
                
                if not browser_results:
                    browser_results.append(f"Browser automation timed out after {timeout_seconds}s - no partial results available")
                
                raise Exception(f"Browser automation timed out after {timeout_seconds}s")
            except Exception as e:
                raise Exception(f"Browser automation execution error: {str(e)}")
            finally:
                # Remove the handler to avoid memory leaks
                browser_logger.removeHandler(ch)
                log_capture_string.close()
            
        except Exception as browser_error:
            logger.error(f"Browser automation error: {browser_error}")
            chunks.append(f"❌ **Browser automation failed:** {str(browser_error)}\n")
            
            # Simple error analysis
            error_str = str(browser_error).lower()
            should_fallback = any(keyword in error_str for keyword in ["timeout", "infinite loop", "unknown", "failed"])
            
            if should_fallback:
                chunks.append("🔄 **Fallback:** Switching to web search\n")
                
                return Command(
                    update={
                        "browser_results": limit_browser_results([f"Browser automation failed: {str(browser_error)} - fallback to web search"]),
                        "streaming_chunks": chunks
                    },
                    goto="web_agent"
                )
            else:
                return Command(
                    update={
                        "browser_results": limit_browser_results([f"Browser automation failed: {str(browser_error)}"]),
                        "streaming_chunks": chunks
                    },
                    goto="evaluator"
                )
    
    except Exception as e:
        logger.error(f"Browser agent error: {e}")
        chunks = [f"❌ **Browser agent error:** {str(e)}\n"]
        
        return Command(
            update={
                "browser_results": limit_browser_results([f"Browser agent failed: {str(e)}"]),
                "streaming_chunks": chunks
            },
            goto="evaluator"
        )