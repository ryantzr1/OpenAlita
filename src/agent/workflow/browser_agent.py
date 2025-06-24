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

from .state import State

logger = logging.getLogger('alita.langgraph')


def browser_agent_router(state: State) -> Literal["evaluator", "web_agent"]:
    """Route browser agent results based on success/failure"""
    mcp_results = state.get("mcp_execution_results", [])
    
    # Check if browser automation failed with loop-related errors
    for result in mcp_results:
        if isinstance(result, str) and any(keyword in result.lower() for keyword in ["infinite loop", "loop", "unknown", "failed", "error"]):
            return "web_agent"  # Fallback to web search only on actual failures
    
    return "evaluator"  # Normal flow - let evaluator decide


def browser_agent_node(state: State) -> Command[Literal["evaluator", "web_agent"]]:
    """Simplified browser agent node that captures browser-use's native logging"""
    logger.info("Browser agent starting...")
    
    # Load environment variables
    load_dotenv()
    
    query = state["original_query"]
    coordinator_analysis = state.get("coordinator_analysis", {})
    browser_capabilities = coordinator_analysis.get("browser_capabilities_needed", [])
    
    chunks = [f"🌐 **Browser Agent:** Starting browser automation\n"]
    chunks.append(f"📝 **Task:** {query}\n")
    
    try:
        # Check if browser-use is available
        try:
            import browser_use
            from browser_use import Agent
        except ImportError:
            chunks.append("❌ **Error:** browser-use not installed\n")
            chunks.append("💡 **Install with:** pip install browser-use\n")
            
            return Command(
                update={
                    "mcp_execution_results": ["Browser automation failed: browser-use not installed"],
                    "streaming_chunks": chunks
                },
                goto="evaluator"
            )
        
        # Check for required API keys
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        if not openai_api_key:
            chunks.append("❌ **Browser automation requires OpenAI API key**\n")
            chunks.append("💡 **Add to .env file:** OPENAI_API_KEY=your_openai_api_key\n")
            
            return Command(
                update={
                    "mcp_execution_results": ["Browser automation failed: OpenAI API key required"],
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
            from langchain_openai import ChatOpenAI
            from browser_use.agent.memory import MemoryConfig

            # Completely disable browser-use cloud features via environment
            os.environ["BROWSER_USE_DISABLE_CLOUD"] = "true"
            os.environ["BROWSER_USE_DISABLE_TELEMETRY"] = "true"

            # Use GPT-4 with optimized settings for browser automation
            chat_model = ChatOpenAI(
                model="gpt-4o",
                openai_api_key=openai_api_key,
                temperature=0,  # Low temperature for consistent actions
                max_tokens=4000,  # More tokens for complex reasoning
            )
            
            planner_llm = ChatOpenAI(model='gpt-4o', openai_api_key=openai_api_key)
            
            chunks.append("🤖 **Using GPT-4**\n")
            
            # Create browser-use agent with minimal configuration
            agent = Agent(
                task=task_description,
                llm=chat_model,
                use_vision=True,
                save_conversation_path=None,
                cloud_sync=False,  # Explicitly disable cloud sync
                is_planner_reasoning=True,
                planner_llm=planner_llm,
                planner_interval=4,
                memory_config=MemoryConfig(
                    llm_instance=chat_model,  
                    agent_id="my_custom_agent",
                    memory_interval=15
                )
            )
            
            chunks.append("✅ **Browser agent configured**\n")
            
            # Execute browser automation
            import asyncio
            import time
            import sys
            import io
            
            # Simple timeout calculation
            timeout_seconds = 60  # Simple default like main branch
            max_steps = 20        # Simple default like main branch
            
            chunks.append(f"🎯 **Executing browser automation...**\n")
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
                # Run browser automation
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
                
                # Use proper browser-use API methods
                try:
                    # Get final result using proper API
                    final_result = result.final_result()
                    if final_result:
                        # Limit size to prevent payload issues
                        result_text = str(final_result)[:1000]
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
                
                # Ensure total results don't get too large
                if len(str(browser_results)) > 10000:  # 10KB limit
                    browser_results = browser_results[:4]  # Keep only first 4 results
                    browser_results.append("Result: Browser automation completed (truncated for size)")
                
                return Command(
                    update={
                        "mcp_execution_results": browser_results,
                        "streaming_chunks": chunks
                    },
                    goto="evaluator"
                )
                
            except asyncio.TimeoutError:
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
                        "mcp_execution_results": [f"Browser automation failed: {str(browser_error)} - fallback to web search"],
                        "streaming_chunks": chunks
                    },
                    goto="web_agent"
                )
            else:
                return Command(
                    update={
                        "mcp_execution_results": [f"Browser automation failed: {str(browser_error)}"],
                        "streaming_chunks": chunks
                    },
                    goto="evaluator"
                )
    
    except Exception as e:
        logger.error(f"Browser agent error: {e}")
        chunks = [f"❌ **Browser agent error:** {str(e)}\n"]
        
        return Command(
            update={
                "mcp_execution_results": [f"Browser agent failed: {str(e)}"],
                "streaming_chunks": chunks
            },
            goto="evaluator"
        )