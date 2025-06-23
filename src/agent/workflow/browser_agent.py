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

from ..llm_provider import LLMProvider
from ..mcp_factory import MCPFactory
from ..mcp_registry import MCPRegistry
from ..prompts import (
    BROWSER_MCP_ANALYSIS_PROMPT,
    BROWSER_TOOL_SCRIPT_PROMPT
)
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
    """Enhanced browser agent node with MCP tool integration"""
    logger.info("Enhanced browser agent with MCP tool integration...")
    
    # Load environment variables
    load_dotenv()
    
    query = state["original_query"]
    coordinator_analysis = state.get("coordinator_analysis", {})
    browser_capabilities = coordinator_analysis.get("browser_capabilities_needed", [])
    
    chunks = [f"🌐 **Enhanced Browser Agent:** Intelligent web automation with MCP integration\n"]
    chunks.append(f"🤖 **Capabilities:** {', '.join(browser_capabilities)}\n")
    chunks.append(f"📝 **Task:** {query}\n")
    
    try:
        # Check if browser-use is available
        try:
            import browser_use
            from browser_use import Agent
        except ImportError:
            chunks.append("❌ **Error:** browser-use not installed\n")
            chunks.append("💡 **Install with:** pip install browser-use\n")
            chunks.append("💡 **Setup browser:** playwright install chromium --with-deps --no-shell\n")
            
            return Command(
                update={
                    "mcp_execution_results": ["Browser automation failed: browser-use not installed"],
                    "streaming_chunks": chunks
                },
                goto="evaluator"
            )
        
        # Initialize MCP components for integration
        mcp_factory = MCPFactory()
        mcp_registry = MCPRegistry()
        llm_provider = LLMProvider()
        
        # Check for required API keys
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        if not openai_api_key:
            chunks.append("❌ **Browser automation requires OpenAI API key**\n")
            chunks.append("💡 **Add to .env file:**\n")
            chunks.append("   OPENAI_API_KEY=your_openai_api_key\n")
            chunks.append("🔍 **Alternative:** Use web search for information queries\n")
            
            return Command(
                update={
                    "mcp_execution_results": ["Browser automation failed: OpenAI API key required"],
                    "streaming_chunks": chunks
                },
                goto="evaluator"
            )
        
        chunks.append("🚀 **Starting enhanced browser automation with MCP integration...**\n")
        
        # Step 1: Pre-browser MCP Tool Analysis
        chunks.append("🧠 **Step 1: Analyzing need for MCP tools during browser automation...**\n")
        
        # Analyze if we need MCP tools during browser automation
        mcp_analysis_prompt = BROWSER_MCP_ANALYSIS_PROMPT.format(query=query)

        # Get MCP analysis
        mcp_analysis_response = ""
        for chunk in llm_provider._make_api_call(mcp_analysis_prompt):
            mcp_analysis_response += chunk
        
        # Parse MCP analysis
        needs_mcp_tools = False
        required_tools = []
        try:
            json_start = mcp_analysis_response.find('{')
            json_end = mcp_analysis_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = mcp_analysis_response[json_start:json_end]
                mcp_analysis = json.loads(json_str)
                needs_mcp_tools = mcp_analysis.get("needs_mcp_tools", False)
                required_tools = mcp_analysis.get("required_tools", [])
                reasoning = mcp_analysis.get("reasoning", "No reasoning provided")
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse MCP analysis: {e}")
            needs_mcp_tools = False
            reasoning = "Fallback: No MCP tools needed"
        
        chunks.append(f"📊 **MCP Analysis:** {reasoning}\n")
        
        # Step 2: Create MCP Tools if Needed
        mcp_tools_created = []
        if needs_mcp_tools and required_tools:
            chunks.append(f"🛠️ **Step 2: Creating {len(required_tools)} MCP tools for browser automation...**\n")
            
            for tool_req in required_tools:
                tool_name = tool_req["name"]
                description = tool_req["description"]
                purpose = tool_req["purpose"]
                timing = tool_req["execution_timing"]
                
                chunks.append(f"🔧 **Creating:** {tool_name} ({timing})\n")
                
                # Generate script for this tool
                script_prompt = BROWSER_TOOL_SCRIPT_PROMPT.format(
                    tool_name=tool_name,
                    description=description,
                    purpose=purpose,
                    query=query,
                    timing=timing
                )

                # Generate script content
                script_content = ""
                for chunk in llm_provider._make_api_call(script_prompt):
                    script_content += chunk
                
                # Create the function
                function, metadata = mcp_factory.create_mcp_from_script(tool_name, script_content)
                
                if function:
                    # Register the tool
                    success = mcp_registry.register_tool(tool_name, function, metadata, script_content)
                    if success:
                        chunks.append(f"   ✅ Registered: {tool_name}\n")
                        mcp_tools_created.append({
                            "name": tool_name,
                            "timing": timing,
                            "description": description
                        })
                    else:
                        chunks.append(f"   ❌ Registration failed: {tool_name}\n")
                else:
                    chunks.append(f"   ❌ Creation failed: {tool_name}\n")
        
        # Step 3: Enhanced Task Description with MCP Integration
        chunks.append("📝 **Step 3: Creating enhanced task description with MCP integration...**\n")
        
        task_analysis = _analyze_browser_task(query)
        enhanced_task = _create_smart_task_description(query, task_analysis)
        
        # Add MCP tool instructions to the task
        if mcp_tools_created:
            mcp_instructions = "\n\nMCP TOOLS AVAILABLE FOR THIS TASK:\n"
            for tool in mcp_tools_created:
                mcp_instructions += f"- {tool['name']}: {tool['description']} (Use: {tool['timing']})\n"
            
            mcp_instructions += """
MCP TOOL INTEGRATION INSTRUCTIONS:
- Use these tools when you need computational assistance
- Call tools before, during, or after browser actions as needed
- Process browser data through these tools when appropriate
- Combine browser automation with computational tools for better results
"""
            enhanced_task += mcp_instructions
        
        chunks.append(f"📋 **Enhanced Task:** {enhanced_task[:150]}...\n")
        
        # Step 4: Configure Enhanced Browser Agent
        chunks.append("⚙️ **Step 4: Configuring enhanced browser agent with MCP integration...**\n")
        
        try:
            from langchain_openai import ChatOpenAI
            from browser_use.agent.memory import MemoryConfig

            
            # Use GPT-4 with optimized settings for browser automation
            chat_model = ChatOpenAI(
                model="gpt-4o",
                openai_api_key=openai_api_key,
                temperature=0,  # Low temperature for consistent actions
                max_tokens=4000,  # More tokens for complex reasoning
            )
            
            chunks.append("🤖 **Using GPT-4 with MCP integration**\n")
            planner_llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=openai_api_key)
            # Create browser-use agent with enhanced configuration
            agent = Agent(
                task=enhanced_task,
                llm=chat_model,
                use_vision=True,  # Enable vision for better element detection
                save_conversation_path=None,  # Disable conversation saving
                cloud_sync=None,  # Disable cloud sync to avoid 404 errors
                is_planner_reasoning=True,
                planner_llm=planner_llm,
                planner_interval=4,
                memory_config=MemoryConfig(
                    llm_instance=chat_model,  
                    agent_id="my_custom_agent",
                    memory_interval=15
                )
            )
            
            chunks.append("✅ **Enhanced browser agent configured successfully**\n")
            
            # Step 5: Execute with MCP Integration
            chunks.append("🎯 **Step 5: Executing with MCP tool integration...**\n")
            
            import asyncio
            import time
            
            # Advanced timeout and monitoring
            timeout_seconds = _calculate_timeout(task_analysis)
            max_steps = _calculate_max_steps(task_analysis)
            
            chunks.append(f"⏱️ **Timeout:** {timeout_seconds}s | **Max Steps:** {max_steps}\n")
            
            # Execute with advanced error handling
            start_time = time.time()
            
            try:
                # Run with comprehensive monitoring
                result = asyncio.run(asyncio.wait_for(
                    agent.run(max_steps=max_steps),
                    timeout=timeout_seconds
                ))
                
                execution_time = time.time() - start_time
                
                chunks.append("✅ **Enhanced browser automation completed successfully!**\n")
                chunks.append(f"⏱️ **Execution Time:** {execution_time:.1f}s\n")
                
                # Step 6: Post-browser MCP Tool Execution
                if mcp_tools_created:
                    chunks.append("🛠️ **Step 6: Executing post-browser MCP tools...**\n")
                    
                    post_browser_tools = [t for t in mcp_tools_created if t["timing"] == "after_browser"]
                    if post_browser_tools:
                        chunks.append(f"📊 **Processing {len(post_browser_tools)} post-browser tools...**\n")
                        
                        for tool in post_browser_tools:
                            try:
                                # Execute post-browser tools with browser results
                                result = mcp_registry.execute_tool(tool["name"], query)
                                chunks.append(f"✅ **{tool['name']}:** {str(result)[:50]}...\n")
                            except Exception as e:
                                chunks.append(f"❌ **{tool['name']} failed:** {str(e)}\n")
                
                # Extract detailed results
                browser_results = _extract_browser_results(result, query, task_analysis, execution_time)
                
                # Add MCP tool information to results
                if mcp_tools_created:
                    browser_results.append(f"MCP tools created and used: {len(mcp_tools_created)}")
                    for tool in mcp_tools_created:
                        browser_results.append(f"- {tool['name']} ({tool['timing']})")
                
                return Command(
                    update={
                        "mcp_execution_results": browser_results,
                        "mcp_tools_created": mcp_tools_created,
                        "streaming_chunks": chunks
                    },
                    goto="evaluator"
                )
                
            except asyncio.TimeoutError:
                raise Exception(f"Browser automation timed out after {timeout_seconds}s (task complexity: {task_analysis['complexity']})")
            except Exception as e:
                raise Exception(f"Browser automation execution error: {str(e)}")
            
        except Exception as browser_error:
            logger.error(f"Enhanced browser automation error: {browser_error}")
            chunks.append(f"❌ **Enhanced browser automation failed:** {str(browser_error)}\n")
            
            # Advanced error analysis and recovery
            recovery_plan = _analyze_browser_error(browser_error, task_analysis)
            chunks.extend(recovery_plan['diagnosis'])
            
            # Intelligent fallback decision
            if recovery_plan['should_fallback_to_web']:
                chunks.append("🔄 **Intelligent Fallback:** Switching to web search\n")
                chunks.append("💡 **Reason:** Browser automation not suitable for this task\n")
                
                return Command(
                    update={
                        "mcp_execution_results": [f"Browser automation failed: {str(browser_error)} - intelligent fallback to web search"],
                        "streaming_chunks": chunks
                    },
                    goto="web_agent"
                )
            else:
                chunks.append("❌ **No suitable fallback available**\n")
                
                return Command(
                    update={
                        "mcp_execution_results": [f"Browser automation failed: {str(browser_error)} - no suitable alternative"],
                        "streaming_chunks": chunks
                    },
                    goto="evaluator"
                )
    
    except Exception as e:
        logger.error(f"Enhanced browser agent error: {e}")
        chunks = [f"❌ **Enhanced browser agent error:** {str(e)}\n"]
        
        return Command(
            update={
                "mcp_execution_results": [f"Enhanced browser agent failed: {str(e)}"],
                "streaming_chunks": chunks
            },
            goto="evaluator"
        )


def _analyze_browser_task(query: str) -> Dict[str, Any]:
    """Intelligent task analysis for browser automation"""
    
    query_lower = query.lower()
    
    # Task type classification
    task_type = "general"
    if any(keyword in query_lower for keyword in ["youtube", "video", "watch", "play"]):
        task_type = "video_platform"
    elif any(keyword in query_lower for keyword in ["login", "sign in", "authenticate"]):
        task_type = "authentication"
    elif any(keyword in query_lower for keyword in ["search", "find", "look up"]):
        task_type = "search"
    elif any(keyword in query_lower for keyword in ["click", "button", "form", "submit"]):
        task_type = "interaction"
    elif any(keyword in query_lower for keyword in ["screenshot", "image", "photo", "visual"]):
        task_type = "visual_analysis"
    elif any(keyword in query_lower for keyword in ["social media", "facebook", "twitter", "instagram", "linkedin"]):
        task_type = "social_media"
    elif any(keyword in query_lower for keyword in ["shopping", "cart", "checkout", "buy", "purchase"]):
        task_type = "ecommerce"
    
    # Complexity assessment
    complexity = "low"
    if len(query.split()) > 15:
        complexity = "high"
    elif len(query.split()) > 8:
        complexity = "medium"
    
    # Required actions analysis
    required_actions = []
    if "navigate" in query_lower or "go to" in query_lower:
        required_actions.append("navigation")
    if "click" in query_lower or "button" in query_lower:
        required_actions.append("clicking")
    if "type" in query_lower or "input" in query_lower or "search" in query_lower:
        required_actions.append("typing")
    if "wait" in query_lower or "load" in query_lower:
        required_actions.append("waiting")
    if "scroll" in query_lower:
        required_actions.append("scrolling")
    if "screenshot" in query_lower or "capture" in query_lower:
        required_actions.append("screenshot")
    
    return {
        "task_type": task_type,
        "complexity": complexity,
        "required_actions": required_actions,
        "word_count": len(query.split()),
        "has_urls": "http" in query_lower,
        "requires_interaction": any(action in ["clicking", "typing", "scrolling"] for action in required_actions)
    }


def _create_smart_task_description(query: str, task_analysis: Dict[str, Any]) -> str:
    """Create intelligent task description based on analysis"""
    
    base_task = f"""
TASK: {query}

TASK ANALYSIS:
- Type: {task_analysis['task_type']}
- Complexity: {task_analysis['complexity']}
- Required Actions: {', '.join(task_analysis['required_actions'])}

CORE INSTRUCTIONS:
1. Navigate to the appropriate website
2. Wait for pages to fully load before proceeding
3. Be patient and methodical in your approach
4. If an action fails, try alternative approaches
5. Provide clear feedback on what you're doing
6. Handle errors gracefully and continue when possible

"""
    
    # Add task-specific instructions
    task_type = task_analysis['task_type']
    
    if task_type == "video_platform":
        base_task += """
VIDEO PLATFORM SPECIFIC:
- Always wait for video player to load completely
- Use Enter key to submit searches if buttons don't work
- Look for video titles, descriptions, and metadata
- Handle different video platform layouts
- Be patient with video loading times
"""
    
    elif task_type == "authentication":
        base_task += """
AUTHENTICATION SPECIFIC:
- Look for login forms carefully
- Wait for page to fully load before entering credentials
- Use Tab key to navigate between fields
- Look for "Sign In", "Login", or "Submit" buttons
- Handle CAPTCHA or 2FA if present
"""
    
    elif task_type == "search":
        base_task += """
SEARCH SPECIFIC:
- Locate search input fields (usually at top of page)
- Click to focus the search field before typing
- Use Enter key to submit searches
- Wait for search results to load completely
- Review results carefully before proceeding
"""
    
    elif task_type == "interaction":
        base_task += """
INTERACTION SPECIFIC:
- Wait for elements to be clickable before clicking
- Look for buttons, links, and interactive elements
- Use alternative methods if primary method fails
- Provide feedback on what you're clicking
- Handle dynamic content that may change
"""
    
    elif task_type == "visual_analysis":
        base_task += """
VISUAL ANALYSIS SPECIFIC:
- Take screenshots when requested
- Analyze images and visual content
- Look for visual elements and their properties
- Handle different image formats and sizes
- Provide detailed visual descriptions
"""
    
    elif task_type == "social_media":
        base_task += """
SOCIAL MEDIA SPECIFIC:
- Handle dynamic feeds and content
- Look for posts, comments, and interactions
- Navigate through different sections carefully
- Handle authentication if required
- Be aware of rate limiting and restrictions
"""
    
    elif task_type == "ecommerce":
        base_task += """
ECOMMERCE SPECIFIC:
- Navigate product pages carefully
- Look for product information, prices, and reviews
- Handle shopping carts and checkout processes
- Look for product images and descriptions
- Handle different e-commerce platform layouts
"""
    
    # Add general best practices
    base_task += """
GENERAL BEST PRACTICES:
- Always wait for pages to load completely
- Use Enter key as alternative to clicking buttons
- Look for multiple ways to accomplish the same task
- Provide clear feedback on your progress
- Handle errors gracefully and continue when possible
- If stuck, try refreshing the page or going back
- Be patient with slow-loading content
"""
    
    return base_task


def _calculate_timeout(task_analysis: Dict[str, Any]) -> int:
    """Calculate appropriate timeout based on task complexity"""
    base_timeout = 30
    
    if task_analysis['complexity'] == "high":
        base_timeout = 60
    elif task_analysis['complexity'] == "medium":
        base_timeout = 45
    
    # Add time for specific actions
    for action in task_analysis['required_actions']:
        if action == "navigation":
            base_timeout += 10
        elif action == "waiting":
            base_timeout += 15
        elif action == "screenshot":
            base_timeout += 5
    
    return min(base_timeout, 120)  # Cap at 2 minutes


def _calculate_max_steps(task_analysis: Dict[str, Any]) -> int:
    """Calculate appropriate max steps based on task complexity"""
    base_steps = 15
    
    if task_analysis['complexity'] == "high":
        base_steps = 25
    elif task_analysis['complexity'] == "medium":
        base_steps = 20
    
    # Add steps for specific actions
    for action in task_analysis['required_actions']:
        if action in ["navigation", "clicking", "typing"]:
            base_steps += 3
        elif action == "waiting":
            base_steps += 2
    
    return min(base_steps, 40)  # Cap at 40 steps


def _extract_browser_results(result, query: str, task_analysis: Dict[str, Any], execution_time: float) -> List[str]:
    """Extract detailed results from browser automation"""
    
    results = [
        f"Enhanced browser automation completed for: {query}",
        f"Task type: {task_analysis['task_type']}",
        f"Complexity: {task_analysis['complexity']}",
        f"Required actions: {', '.join(task_analysis['required_actions'])}",
        f"Execution time: {execution_time:.1f}s",
        f"LLM used: GPT-4",
    ]
    
    # Add result-specific information
    if hasattr(result, 'steps'):
        results.append(f"Steps executed: {len(result.steps)}")
    
    # Extract the actual answer from browser automation
    if hasattr(result, 'final_answer'):
        results.append(f"Browser Answer: {result.final_answer}")
    elif hasattr(result, 'result'):
        results.append(f"Browser Answer: {result.result}")
    elif hasattr(result, 'answer'):
        results.append(f"Browser Answer: {result.answer}")
    
    # Also check for any text content that might contain the answer
    if hasattr(result, 'text') and result.text:
        results.append(f"Browser Text: {result.text}")
    
    # Check for any files or attachments that might contain results
    if hasattr(result, 'files_to_display') and result.files_to_display:
        for file_info in result.files_to_display:
            if isinstance(file_info, str) and 'results.md' in file_info:
                results.append(f"Results File: {file_info}")
    
    results.append("Result: Enhanced browser automation completed successfully")
    
    return results


def _analyze_browser_error(error: Exception, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Intelligent error analysis and recovery planning"""
    
    error_str = str(error).lower()
    
    diagnosis = []
    should_fallback_to_web = False
    
    # Analyze error patterns
    if "infinite loop" in error_str or "loop" in error_str:
        diagnosis.extend([
            "🔄 **Issue: Infinite Loop Detected**",
            "**Root Causes:**",
            "   - Repeated actions without progress",
            "   - Element detection failures",
            "   - Navigation stuck on same page",
            "   - Unknown actions from LLM",
            "**Advanced Solutions Applied:**",
            "   - Intelligent step limit enforcement",
            "   - Action repetition detection",
            "   - URL navigation monitoring",
            "   - Unknown action detection",
            "**Recommendations:**",
            "   - Try simpler, more specific tasks",
            "   - Use web search for information queries",
            "   - Break complex tasks into smaller steps"
        ])
        should_fallback_to_web = True
    
    elif "unknown()" in error_str or "unknown" in error_str:
        diagnosis.extend([
            "🔍 **Issue: Unknown Action Detected**",
            "**Root Causes:**",
            "   - LLM not returning properly formatted actions",
            "   - Element detection failures on dynamic pages",
            "   - API response format issues",
            "**Advanced Solutions Applied:**",
            "   - Enhanced message handling",
            "   - Structured fallback responses",
            "   - Better error recovery",
            "   - Alternative action mapping"
        ])
        should_fallback_to_web = True
    
    elif "timeout" in error_str:
        diagnosis.extend([
            "⏰ **Issue: Timeout**",
            f"   - Task took too long to complete (complexity: {task_analysis['complexity']})",
            "   - Consider simplifying the task",
            "   - Try breaking into smaller subtasks"
        ])
        should_fallback_to_web = True
    
    elif "api" in error_str or "connection" in error_str:
        diagnosis.extend([
            "🔗 **Issue: API Connection**",
            "**Root Causes:**",
            "   - Invalid or expired OpenAI API key",
            "   - Network connectivity issues",
            "   - OpenAI API service unavailable",
            "   - Rate limiting or quota exceeded",
            "**Solutions:**",
            "   - Verify OPENAI_API_KEY in .env file",
            "   - Check network connection",
            "   - Ensure OpenAI API key has sufficient credits",
            "   - Try again later if rate limited"
        ])
        should_fallback_to_web = False  # API issues don't warrant web fallback
    
    else:
        diagnosis.extend([
            "💡 **General Advanced Troubleshooting:**",
            "   - Verify browser-use installation and setup",
            "   - Check Playwright browser installation",
            "   - Try simpler automation tasks first",
            "   - Review task complexity and requirements",
            "   - Consider alternative approaches"
        ])
        should_fallback_to_web = task_analysis['task_type'] in ['search', 'general']
    
    return {
        "diagnosis": diagnosis,
        "should_fallback_to_web": should_fallback_to_web,
        "error_type": "unknown" if "unknown" in error_str else "timeout" if "timeout" in error_str else "api" if "api" in error_str else "general"
    } 