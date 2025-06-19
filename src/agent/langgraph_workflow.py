# LangGraph Multi-Agent Workflow System for Alita
# This module provides graph-based orchestration of specialized agents

import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence
from enum import Enum
import json
import asyncio
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from .llm_provider import LLMProvider
from .mcp_box import MCPBox
from .web_agent import WebAgent

# Enhanced logger setup for detailed workflow tracking
logger = logging.getLogger('alita.langgraph')
workflow_logger = logging.getLogger('alita.workflow')

class WorkflowLogger:
    """Dedicated logger for tracking workflow execution steps."""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.step_counter = 0
        self.start_time = datetime.now()
        
    def log_step(self, agent_name: str, action: str, details: Dict[str, Any] = None, status: str = "success"):
        """Log a detailed workflow step."""
        self.step_counter += 1
        timestamp = datetime.now()
        elapsed = (timestamp - self.start_time).total_seconds()
        
        log_entry = {
            "session_id": self.session_id,
            "step": self.step_counter,
            "timestamp": timestamp.isoformat(),
            "elapsed_seconds": elapsed,
            "agent": agent_name,
            "action": action,
            "status": status,
            "details": details or {}
        }
        
        # Log with structured format
        workflow_logger.info(f"[STEP {self.step_counter}] {agent_name} -> {action} ({status}) | {json.dumps(details or {}, indent=None)}")
        
        return log_entry
    
    def log_workflow_start(self, query: str):
        """Log the start of a workflow."""
        workflow_logger.info(f"=== WORKFLOW START [Session: {self.session_id}] ===")
        workflow_logger.info(f"Query: {query}")
        
    def log_workflow_end(self, success: bool, final_result: str = None):
        """Log the end of a workflow."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        status = "SUCCESS" if success else "FAILED"
        workflow_logger.info(f"=== WORKFLOW END [Session: {self.session_id}] - {status} in {elapsed:.2f}s ===")
        if final_result:
            workflow_logger.info(f"Final Result: {final_result[:200]}...")

class AgentRole(Enum):
    """Defines the roles that agents can take in the workflow."""
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"
    VALIDATOR = "validator"
    EXECUTOR = "executor"

class TaskComplexity(Enum):
    """Defines task complexity levels for routing decisions."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_STEP = "multi_step"

class WorkflowState(TypedDict):
    """State object that flows through the graph nodes."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_query: str
    task_complexity: TaskComplexity
    current_step: int
    max_steps: int
    context: Dict[str, Any]
    results: Dict[str, Any]
    agent_outputs: Dict[str, Any]
    next_action: Optional[str]
    is_complete: bool
    error_state: Optional[str]
    workflow_logger: WorkflowLogger  # Add logger to state

class SpecializedAgent:
    """Base class for specialized agents in the workflow."""
    
    def __init__(self, role: AgentRole, llm_provider: LLMProvider, name: str = None):
        self.role = role
        self.llm_provider = llm_provider
        self.name = name or role.value
        self.capabilities = []
        self.tools = []
        
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Process the current state and return updated state."""
        raise NotImplementedError("Subclasses must implement process method")
    
    def can_handle(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if this agent can handle the given query."""
        return True

class OrchestratorAgent(SpecializedAgent):
    """Orchestrates the entire workflow and makes routing decisions."""
    
    def __init__(self, llm_provider: LLMProvider, mcp_box: MCPBox):
        super().__init__(AgentRole.ORCHESTRATOR, llm_provider, "workflow_orchestrator")
        self.mcp_box = mcp_box
        self.capabilities = ["task_analysis", "workflow_planning", "agent_routing"]
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Analyze the query and plan the workflow."""
        wf_logger = state.get('workflow_logger')
        
        logger.info(f"Orchestrator processing query: {state['original_query']}")
        wf_logger.log_step("orchestrator", "start_analysis", {"query": state['original_query']})
        
        # Analyze task complexity
        wf_logger.log_step("orchestrator", "analyzing_complexity")
        complexity = await self._analyze_task_complexity(state['original_query'])
        state['task_complexity'] = complexity
        wf_logger.log_step("orchestrator", "complexity_determined", {"complexity": complexity.value})
        
        # Plan workflow steps
        wf_logger.log_step("orchestrator", "creating_workflow_plan")
        workflow_plan = await self._create_workflow_plan(state)
        state['context']['workflow_plan'] = workflow_plan
        wf_logger.log_step("orchestrator", "workflow_plan_created", {
            "plan_steps": len(workflow_plan),
            "plan": workflow_plan
        })
        
        # Determine next action
        state['next_action'] = self._determine_next_action(state)
        wf_logger.log_step("orchestrator", "next_action_determined", {"next_action": state['next_action']})
        
        # Add orchestrator message
        orchestrator_msg = AIMessage(
            content=f"🤖 Orchestrator: Analyzed task complexity as {complexity.value}. "
                   f"Planned {len(workflow_plan)} workflow steps. Next: {state['next_action']}",
            name=self.name
        )
        state['messages'].append(orchestrator_msg)
        
        wf_logger.log_step("orchestrator", "process_complete", {"message_added": True})
        return state
    
    async def _analyze_task_complexity(self, query: str) -> TaskComplexity:
        """Analyze the complexity of the incoming task."""
        complexity_prompt = f"""
        Analyze the complexity of this user query and classify it:
        
        Query: "{query}"
        
        Classification criteria:
        - SIMPLE: Single, direct questions or basic commands
        - MODERATE: Questions requiring some research or calculation
        - COMPLEX: Multi-faceted questions requiring deep analysis
        - MULTI_STEP: Tasks requiring multiple sequential operations
        
        Respond with only one word: SIMPLE, MODERATE, COMPLEX, or MULTI_STEP
        """
        
        try:
            response_chunks = []
            for chunk in self.llm_provider._make_api_call(complexity_prompt):
                if not chunk.startswith("Error:"):
                    response_chunks.append(chunk)
            
            complexity_str = "".join(response_chunks).strip().upper()
            logger.info(f"LLM classified complexity as: {complexity_str}")
            
            for complexity in TaskComplexity:
                if complexity.name == complexity_str:
                    return complexity
            
            # Default to MODERATE if classification fails
            logger.warning(f"Unknown complexity classification: {complexity_str}, defaulting to MODERATE")
            return TaskComplexity.MODERATE
            
        except Exception as e:
            logger.warning(f"Error analyzing task complexity: {e}")
            return TaskComplexity.MODERATE
    
    async def _create_workflow_plan(self, state: WorkflowState) -> List[Dict[str, Any]]:
        """Create a detailed workflow plan based on the query and complexity."""
        query = state['original_query']
        complexity = state['task_complexity']
        
        planning_prompt = f"""
        Create a workflow plan for this query based on its complexity level:
        
        Query: "{query}"
        Complexity: {complexity.value}
        
        Available agent types:
        - researcher: Gathers information from various sources
        - analyzer: Analyzes data and extracts insights
        - synthesizer: Combines information into coherent responses
        - validator: Validates results and checks accuracy
        - executor: Executes specific tasks or commands
        
        Create a JSON array of workflow steps, each with:
        - agent: which agent type to use
        - task: specific task description
        - dependencies: list of previous steps this depends on
        - estimated_duration: rough time estimate
        
        Example format:
        [
            {{"agent": "researcher", "task": "gather information", "dependencies": [], "estimated_duration": "short"}},
            {{"agent": "analyzer", "task": "analyze findings", "dependencies": ["researcher"], "estimated_duration": "medium"}}
        ]
        
        Respond with only the JSON array:
        """
        
        try:
            response_chunks = []
            for chunk in self.llm_provider._make_api_call(planning_prompt):
                if not chunk.startswith("Error:"):
                    response_chunks.append(chunk)
            
            plan_text = "".join(response_chunks).strip()
            plan = json.loads(plan_text)
            
            return plan if isinstance(plan, list) else []
            
        except Exception as e:
            logger.warning(f"Error creating workflow plan: {e}")
            # Fallback plan
            return [
                {"agent": "researcher", "task": "gather information", "dependencies": [], "estimated_duration": "short"},
                {"agent": "synthesizer", "task": "provide response", "dependencies": ["researcher"], "estimated_duration": "short"}
            ]
    
    def _determine_next_action(self, state: WorkflowState) -> str:
        """Determine the next action based on current state."""
        workflow_plan = state['context'].get('workflow_plan', [])
        current_step = state['current_step']
        
        if current_step < len(workflow_plan):
            next_step = workflow_plan[current_step]
            return next_step['agent']
        
        return "synthesizer"  # Default to synthesis if plan is complete

class ResearcherAgent(SpecializedAgent):
    """Specialized in gathering information from various sources."""
    
    def __init__(self, llm_provider: LLMProvider, web_agent: WebAgent, mcp_box: MCPBox):
        super().__init__(AgentRole.RESEARCHER, llm_provider, "research_specialist")
        self.web_agent = web_agent
        self.mcp_box = mcp_box
        self.capabilities = ["web_search", "data_gathering", "source_verification"]
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Gather comprehensive information about the query."""
        wf_logger = state.get('workflow_logger')
        query = state['original_query']
        
        logger.info(f"Researcher gathering information for: {query}")
        wf_logger.log_step("researcher", "start_research", {"query": query})
        
        research_results = {}
        
        # Try web search first
        if self.web_agent.can_handle_with_search(query):
            wf_logger.log_step("researcher", "attempting_web_search")
            logger.info("Performing web search research")
            try:
                web_results = self.web_agent.answer_query(query)
                if web_results:
                    research_results['web_search'] = web_results
                    wf_logger.log_step("researcher", "web_search_success", {
                        "result_length": len(web_results),
                        "preview": web_results[:100] + "..." if len(web_results) > 100 else web_results
                    })
                else:
                    wf_logger.log_step("researcher", "web_search_no_results")
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
                wf_logger.log_step("researcher", "web_search_failed", {"error": str(e)}, "error")
        else:
            wf_logger.log_step("researcher", "web_search_skipped", {"reason": "query not suitable for web search"})
        
        # Check for relevant MCPs
        wf_logger.log_step("researcher", "searching_relevant_mcps")
        relevant_mcps = self._find_relevant_mcps(query)
        if relevant_mcps:
            logger.info(f"Found {len(relevant_mcps)} relevant MCPs")
            research_results['mcp_capabilities'] = [
                {"name": name, "description": mcp['description']} 
                for name, mcp in relevant_mcps.items()
            ]
            wf_logger.log_step("researcher", "mcps_found", {
                "count": len(relevant_mcps),
                "mcp_names": list(relevant_mcps.keys())
            })
        else:
            wf_logger.log_step("researcher", "no_mcps_found")
        
        # Store research results
        state['agent_outputs']['researcher'] = research_results
        state['results']['research_data'] = research_results
        
        # Add researcher message
        research_msg = AIMessage(
            content=f"🔍 Researcher: Gathered information from {len(research_results)} sources. "
                   f"Found {'web data' if 'web_search' in research_results else 'no web data'} "
                   f"and {len(relevant_mcps)} relevant tools.",
            name=self.name
        )
        state['messages'].append(research_msg)
        
        wf_logger.log_step("researcher", "research_complete", {
            "sources_found": len(research_results),
            "has_web_data": 'web_search' in research_results,
            "mcp_count": len(relevant_mcps)
        })
        
        return state
    
    def _find_relevant_mcps(self, query: str) -> Dict[str, Any]:
        """Find MCPs relevant to the current query."""
        relevant_mcps = {}
        query_lower = query.lower()
        
        for mcp_name, mcp_data in self.mcp_box.mcps.items():
            description = mcp_data.get('description', '').lower()
            
            # Simple keyword matching - could be enhanced with LLM-based similarity
            if any(word in description for word in query_lower.split() if len(word) > 3):
                relevant_mcps[mcp_name] = mcp_data
        
        return relevant_mcps

class AnalyzerAgent(SpecializedAgent):
    """Specialized in analyzing data and extracting insights."""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__(AgentRole.ANALYZER, llm_provider, "data_analyzer")
        self.capabilities = ["data_analysis", "pattern_recognition", "insight_extraction"]
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Analyze the gathered research data."""
        wf_logger = state.get('workflow_logger')
        
        logger.info("Analyzer processing research data")
        wf_logger.log_step("analyzer", "start_analysis")
        
        research_data = state['results'].get('research_data', {})
        
        if not research_data:
            wf_logger.log_step("analyzer", "no_data_to_analyze", status="warning")
            state['agent_outputs']['analyzer'] = {"status": "no_data", "message": "No research data to analyze"}
            return state
        
        wf_logger.log_step("analyzer", "analyzing_data", {"data_sources": list(research_data.keys())})
        
        # Perform analysis
        analysis_results = await self._analyze_research_data(research_data, state['original_query'])
        
        state['agent_outputs']['analyzer'] = analysis_results
        state['results']['analysis'] = analysis_results
        
        # Add analyzer message
        insights_count = len(analysis_results.get('insights', []))
        analysis_msg = AIMessage(
            content=f"📊 Analyzer: Completed analysis. Found {insights_count} key insights. "
                   f"Confidence: {analysis_results.get('confidence_level', 'unknown')}",
            name=self.name
        )
        state['messages'].append(analysis_msg)
        
        wf_logger.log_step("analyzer", "analysis_complete", {
            "insights_found": insights_count,
            "confidence": analysis_results.get('confidence_level'),
            "key_findings_count": len(analysis_results.get('key_findings', []))
        })
        
        return state
    
    async def _analyze_research_data(self, research_data: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Analyze the research data using LLM."""
        analysis_prompt = f"""
        Analyze the following research data for the query: "{original_query}"
        
        Research Data:
        {json.dumps(research_data, indent=2)}
        
        Provide analysis in JSON format with:
        - key_findings: List of main discoveries
        - insights: List of important insights
        - confidence_level: How confident you are in the analysis (high/medium/low)
        - recommendations: Suggested next steps
        - data_quality: Assessment of data quality and completeness
        
        Respond with only the JSON object:
        """
        
        try:
            response_chunks = []
            for chunk in self.llm_provider._make_api_call(analysis_prompt):
                if not chunk.startswith("Error:"):
                    response_chunks.append(chunk)
            
            analysis_text = "".join(response_chunks).strip()
            analysis = json.loads(analysis_text)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error in data analysis: {e}")
            return {
                "key_findings": ["Analysis failed due to technical error"],
                "insights": [],
                "confidence_level": "low",
                "recommendations": ["Retry with simpler approach"],
                "data_quality": "unknown"
            }

class SynthesizerAgent(SpecializedAgent):
    """Specialized in synthesizing information into coherent responses."""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__(AgentRole.SYNTHESIZER, llm_provider, "response_synthesizer")
        self.capabilities = ["information_synthesis", "response_generation", "narrative_creation"]
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Synthesize all gathered information into a final response."""
        wf_logger = state.get('workflow_logger')
        
        logger.info("Synthesizer creating final response")
        wf_logger.log_step("synthesizer", "start_synthesis")
        
        # Gather all available data
        research_data = state['results'].get('research_data', {})
        analysis_data = state['results'].get('analysis', {})
        execution_data = state['results'].get('execution', {})
        
        wf_logger.log_step("synthesizer", "gathering_data", {
            "has_research": bool(research_data),
            "has_analysis": bool(analysis_data),
            "has_execution": bool(execution_data)
        })
        
        # Create synthesized response
        final_response = await self._synthesize_response(
            state['original_query'], 
            research_data, 
            analysis_data,
            execution_data
        )
        
        state['agent_outputs']['synthesizer'] = {"final_response": final_response}
        state['results']['final_answer'] = final_response
        state['is_complete'] = True
        
        # Add synthesizer message
        synthesis_msg = AIMessage(
            content=final_response,
            name=self.name
        )
        state['messages'].append(synthesis_msg)
        
        wf_logger.log_step("synthesizer", "synthesis_complete", {
            "response_length": len(final_response),
            "response_preview": final_response[:100] + "..." if len(final_response) > 100 else final_response
        })
        
        return state
    
    async def _synthesize_response(self, query: str, research_data: Dict[str, Any], 
                                 analysis_data: Dict[str, Any], execution_data: Dict[str, Any] = None) -> str:
        """Synthesize a comprehensive response."""
        synthesis_prompt = f"""
        Create a comprehensive, well-structured response to this user query:
        
        Query: "{query}"
        
        Available Information:
        Research Data: {json.dumps(research_data, indent=2)}
        Analysis: {json.dumps(analysis_data, indent=2)}
        Execution Results: {json.dumps(execution_data or {}, indent=2)}
        
        Requirements:
        1. Directly answer the user's question
        2. Include relevant supporting information
        3. Be clear and well-organized
        4. Acknowledge limitations if data is incomplete
        5. Use a natural, conversational tone
        
        Provide the final response:
        """
        
        try:
            response_chunks = []
            for chunk in self.llm_provider._make_api_call(synthesis_prompt):
                if not chunk.startswith("Error:"):
                    response_chunks.append(chunk)
            
            return "".join(response_chunks).strip()
            
        except Exception as e:
            logger.error(f"Error in response synthesis: {e}")
            return f"I apologize, but I encountered an error while synthesizing the response to your query: '{query}'. Please try rephrasing your question."

class ExecutorAgent(SpecializedAgent):
    """Specialized in executing specific tasks and commands."""
    
    def __init__(self, llm_provider: LLMProvider, mcp_box: MCPBox, mcp_factory):
        super().__init__(AgentRole.EXECUTOR, llm_provider, "task_executor")
        self.mcp_box = mcp_box
        self.mcp_factory = mcp_factory
        self.capabilities = ["task_execution", "mcp_creation", "command_processing"]
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Execute specific tasks or create MCPs as needed."""
        wf_logger = state.get('workflow_logger')
        query = state['original_query']
        
        logger.info(f"Executor processing task for: {query}")
        wf_logger.log_step("executor", "start_execution", {"query": query})
        
        execution_results = {}
        
        # Check if we need to create or execute an MCP
        wf_logger.log_step("executor", "searching_executable_mcps")
        relevant_mcps = self._find_executable_mcps(query)
        
        if relevant_mcps:
            # Execute existing MCP
            mcp_name = list(relevant_mcps.keys())[0]
            logger.info(f"Executing existing MCP: {mcp_name}")
            wf_logger.log_step("executor", "executing_existing_mcp", {"mcp_name": mcp_name})
            
            try:
                # Set up command context like in the original agent
                import builtins
                original_command = getattr(builtins, '_current_user_command', None)
                builtins._current_user_command = query
                
                try:
                    mcp_output = relevant_mcps[mcp_name]["function"]()
                    execution_results['mcp_execution'] = {
                        'mcp_name': mcp_name,
                        'output': mcp_output,
                        'status': 'success'
                    }
                    wf_logger.log_step("executor", "mcp_execution_success", {
                        "mcp_name": mcp_name,
                        "output_length": len(str(mcp_output)) if mcp_output else 0
                    })
                finally:
                    # Restore original command context
                    if original_command is not None:
                        builtins._current_user_command = original_command
                    elif hasattr(builtins, '_current_user_command'):
                        delattr(builtins, '_current_user_command')
                        
            except Exception as e:
                execution_results['mcp_execution'] = {
                    'mcp_name': mcp_name,
                    'error': str(e),
                    'status': 'failed'
                }
                wf_logger.log_step("executor", "mcp_execution_failed", {
                    "mcp_name": mcp_name,
                    "error": str(e)
                }, "error")
        else:
            # Need to create a new MCP
            logger.info("Creating new MCP for execution")
            wf_logger.log_step("executor", "creating_new_mcp")
            
            try:
                mcp_creation_result = await self._create_and_execute_mcp(query)
                execution_results['mcp_creation'] = mcp_creation_result
                
                if mcp_creation_result.get('status') == 'success':
                    wf_logger.log_step("executor", "mcp_creation_success", {
                        "mcp_name": mcp_creation_result.get('mcp_name'),
                        "output_available": 'output' in mcp_creation_result
                    })
                else:
                    wf_logger.log_step("executor", "mcp_creation_failed", {
                        "error": mcp_creation_result.get('error')
                    }, "error")
                    
            except Exception as e:
                execution_results['mcp_creation'] = {
                    'error': str(e),
                    'status': 'failed'
                }
                wf_logger.log_step("executor", "mcp_creation_exception", {"error": str(e)}, "error")
        
        state['agent_outputs']['executor'] = execution_results
        state['results']['execution'] = execution_results
        
        # Add executor message
        if 'mcp_execution' in execution_results:
            status = execution_results['mcp_execution']['status']
            mcp_name = execution_results['mcp_execution']['mcp_name']
            executor_msg = AIMessage(
                content=f"⚙️ Executor: {'Successfully executed' if status == 'success' else 'Failed to execute'} MCP '{mcp_name}'.",
                name=self.name
            )
        else:
            status = execution_results.get('mcp_creation', {}).get('status', 'unknown')
            executor_msg = AIMessage(
                content=f"⚙️ Executor: {'Successfully created and executed' if status == 'success' else 'Failed to create'} new MCP.",
                name=self.name
            )
        
        state['messages'].append(executor_msg)
        
        wf_logger.log_step("executor", "execution_complete", {
            "final_status": status,
            "execution_type": "existing_mcp" if 'mcp_execution' in execution_results else "new_mcp"
        })
        
        return state
    
    def _find_executable_mcps(self, query: str) -> Dict[str, Any]:
        """Find MCPs that can execute the given query."""
        executable_mcps = {}
        query_lower = query.lower()
        
        for mcp_name, mcp_data in self.mcp_box.mcps.items():
            # Check if MCP seems relevant to the query
            description = mcp_data.get('description', '').lower()
            if any(word in description for word in query_lower.split() if len(word) > 3):
                executable_mcps[mcp_name] = mcp_data
        
        return executable_mcps
    
    async def _create_and_execute_mcp(self, query: str) -> Dict[str, Any]:
        """Create a new MCP and execute it."""
        try:
            # Generate MCP name
            tokens = query.split()
            mcp_name = self._generate_mcp_name(query, tokens)
            
            # Generate MCP script
            final_script = None
            for chunk in self.llm_provider.generate_mcp_script_streaming(mcp_name, query):
                if isinstance(chunk, str) and not chunk.startswith("Error:"):
                    continue  # Skip streaming chunks for now
            
            final_script = self.llm_provider.get_last_generated_mcp_script()
            
            if not final_script:
                return {"status": "failed", "error": "Failed to generate MCP script"}
            
            # Create MCP function
            mcp_function, mcp_metadata = self.mcp_factory.create_mcp_from_script(mcp_name, final_script)
            
            if not mcp_function:
                return {"status": "failed", "error": "Failed to create MCP function"}
            
            # Register the MCP
            mcp_metadata['original_command'] = query
            self.mcp_box.register_mcp(
                name=mcp_name,
                function=mcp_function,
                metadata=mcp_metadata
            )
            
            # Execute the new MCP
            import builtins
            original_command = getattr(builtins, '_current_user_command', None)
            builtins._current_user_command = query
            
            try:
                mcp_output = mcp_function()
                return {
                    "status": "success",
                    "mcp_name": mcp_name,
                    "output": mcp_output
                }
            finally:
                if original_command is not None:
                    builtins._current_user_command = original_command
                elif hasattr(builtins, '_current_user_command'):
                    delattr(builtins, '_current_user_command')
                    
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def _generate_mcp_name(self, command_str: str, tokens: List[str]) -> str:
        """Generate a simple MCP name."""
        meaningful_words = [word.lower() for word in tokens[:3] if len(word) > 2]
        if meaningful_words:
            return '_'.join(meaningful_words[:2]).replace('-', '_').replace('.', '_')
        else:
            return 'custom_tool'


class LangGraphWorkflowManager:
    """Main manager for LangGraph-based multi-agent workflows."""
    
    def __init__(self, llm_provider: LLMProvider, mcp_box: MCPBox, web_agent: WebAgent, mcp_factory):
        self.llm_provider = llm_provider
        self.mcp_box = mcp_box
        self.web_agent = web_agent
        self.mcp_factory = mcp_factory
        
        # Initialize specialized agents
        self.orchestrator = OrchestratorAgent(llm_provider, mcp_box)
        self.researcher = ResearcherAgent(llm_provider, web_agent, mcp_box)
        self.analyzer = AnalyzerAgent(llm_provider)
        self.synthesizer = SynthesizerAgent(llm_provider)
        self.executor = ExecutorAgent(llm_provider, mcp_box, mcp_factory)
        
        # Create the workflow graph
        self.workflow = self._create_workflow_graph()
        
        logger.info("LangGraph Workflow Manager initialized with all specialized agents")
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow graph."""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        
        # Define the workflow edges
        workflow.set_entry_point("orchestrator")
        
        # Conditional routing from orchestrator
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_after_orchestrator,
            {
                "researcher": "researcher",
                "executor": "executor",
                "synthesizer": "synthesizer"
            }
        )
        
        # Research to analysis or synthesis
        workflow.add_conditional_edges(
            "researcher",
            self._route_after_research,
            {
                "analyzer": "analyzer",
                "synthesizer": "synthesizer",
                "executor": "executor"
            }
        )
        
        # Analysis to synthesis
        workflow.add_edge("analyzer", "synthesizer")
        
        # Execution to synthesis
        workflow.add_edge("executor", "synthesizer")
        
        # Synthesis to end
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    async def _orchestrator_node(self, state: WorkflowState) -> WorkflowState:
        """Process state through orchestrator agent."""
        return await self.orchestrator.process(state)
    
    async def _researcher_node(self, state: WorkflowState) -> WorkflowState:
        """Process state through researcher agent."""
        return await self.researcher.process(state)
    
    async def _analyzer_node(self, state: WorkflowState) -> WorkflowState:
        """Process state through analyzer agent."""
        return await self.analyzer.process(state)
    
    async def _executor_node(self, state: WorkflowState) -> WorkflowState:
        """Process state through executor agent."""
        return await self.executor.process(state)
    
    async def _synthesizer_node(self, state: WorkflowState) -> WorkflowState:
        """Process state through synthesizer agent."""
        return await self.synthesizer.process(state)
    
    def _route_after_orchestrator(self, state: WorkflowState) -> str:
        """Route after orchestrator based on task complexity and requirements."""
        complexity = state.get('task_complexity', TaskComplexity.MODERATE)
        query = state['original_query'].lower()
        
        # Check if this needs direct execution (commands, calculations, etc.)
        execution_keywords = ['calculate', 'compute', 'create', 'generate', 'execute', 'run']
        if any(keyword in query for keyword in execution_keywords):
            return "executor"
        
        # For simple tasks, go straight to synthesis
        if complexity == TaskComplexity.SIMPLE:
            return "synthesizer"
        
        # For moderate to complex tasks, start with research
        return "researcher"
    
    def _route_after_research(self, state: WorkflowState) -> str:
        """Route after research based on findings and complexity."""
        complexity = state.get('task_complexity', TaskComplexity.MODERATE)
        research_data = state['results'].get('research_data', {})
        
        # If we found MCPs that need execution, go to executor
        if 'mcp_capabilities' in research_data and research_data['mcp_capabilities']:
            return "executor"
        
        # For complex tasks with substantial data, analyze first
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.MULTI_STEP] and research_data:
            return "analyzer"
        
        # Otherwise go straight to synthesis
        return "synthesizer"
    
    async def process_query_streaming(self, query: str) -> List[str]:
        """Process a query through the LangGraph workflow with streaming updates."""
        # Create workflow logger for this session
        wf_logger = WorkflowLogger()
        wf_logger.log_workflow_start(query)
        
        logger.info(f"LangGraph workflow processing query: {query}")
        
        # Initialize state with logger
        initial_state = WorkflowState(
            messages=[HumanMessage(content=query)],
            original_query=query,
            task_complexity=TaskComplexity.MODERATE,  # Will be updated by orchestrator
            current_step=0,
            max_steps=10,
            context={},
            results={},
            agent_outputs={},
            next_action=None,
            is_complete=False,
            error_state=None,
            workflow_logger=wf_logger
        )
        
        output_messages = []
        workflow_success = False
        
        try:
            # Execute the workflow
            async for output in self.workflow.astream(initial_state):
                for node_name, state in output.items():
                    if node_name != "__end__":
                        wf_logger.log_step("workflow_manager", f"node_{node_name}_completed", {
                            "node": node_name,
                            "state_keys": list(state.keys())
                        })
                        
                        # Get the latest message from this agent
                        if state.get('messages'):
                            latest_message = state['messages'][-1]
                            if isinstance(latest_message, AIMessage) and latest_message.content:
                                output_messages.append(latest_message.content)
                                logger.info(f"Node {node_name} completed: {latest_message.content[:100]}...")
            
            # Get final result
            final_state = None
            async for output in self.workflow.astream(initial_state):
                for node_name, state in output.items():
                    if node_name == "__end__":
                        final_state = state
                        break
                if final_state:
                    break
            
            if final_state and final_state.get('results', {}).get('final_answer'):
                final_answer = final_state['results']['final_answer']
                if final_answer not in output_messages:
                    output_messages.append(final_answer)
                workflow_success = True
                wf_logger.log_workflow_end(True, final_answer)
            else:
                wf_logger.log_workflow_end(False, "No final answer generated")
            
        except Exception as e:
            error_msg = f"Error in LangGraph workflow: {str(e)}"
            logger.error(error_msg, exc_info=True)
            wf_logger.log_step("workflow_manager", "workflow_error", {"error": str(e)}, "error")
            wf_logger.log_workflow_end(False, error_msg)
            output_messages.append(error_msg)
        
        return output_messages
    
    def should_use_langgraph(self, query: str) -> bool:
        """Determine if a query should use the LangGraph workflow."""
        query_lower = query.lower()
        
        # Use LangGraph for complex research queries
        research_indicators = [
            'analyze', 'compare', 'research', 'investigate', 'study',
            'what is', 'how does', 'why', 'explain', 'tell me about'
        ]
        
        # Use LangGraph for multi-step tasks
        multi_step_indicators = [
            'first', 'then', 'after', 'next', 'finally',
            'step by step', 'process', 'workflow'
        ]
        
        # Use LangGraph for data analysis tasks
        analysis_indicators = [
            'statistics', 'trends', 'patterns', 'insights',
            'correlation', 'relationship', 'impact'
        ]
        
        all_indicators = research_indicators + multi_step_indicators + analysis_indicators
        
        # Check if query contains multiple indicators or is sufficiently complex
        indicator_count = sum(1 for indicator in all_indicators if indicator in query_lower)
        
        return indicator_count >= 1 or len(query.split()) > 8