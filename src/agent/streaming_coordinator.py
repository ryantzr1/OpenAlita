"""
Streaming Coordinator for Real-time Frontend Updates

This module provides advanced streaming capabilities for the LangGraph workflow,
enabling real-time updates to the frontend with proper state management.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Generator, Optional
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger('alita.streaming')

class StreamEventType(Enum):
    """Types of streaming events"""
    WORKFLOW_START = "workflow_start"
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    TOOL_CREATED = "tool_created"
    TOOL_EXECUTED = "tool_executed"
    WEB_SEARCH_RESULTS = "web_search_results"
    EVALUATION_UPDATE = "evaluation_update"
    SYNTHESIS_PROGRESS = "synthesis_progress"
    WORKFLOW_COMPLETE = "workflow_complete"
    ERROR = "error"

@dataclass
class StreamEvent:
    """Event structure for streaming updates"""
    event_type: StreamEventType
    agent_name: str
    timestamp: float
    data: Dict[str, Any]
    session_id: str
    step_id: str
    
    def to_json(self) -> str:
        """Convert event to JSON string for streaming"""
        return json.dumps({
            'event_type': self.event_type.value,
            'agent_name': self.agent_name,
            'timestamp': self.timestamp,
            'data': self.data,
            'session_id': self.session_id,
            'step_id': self.step_id
        })

class StreamingCoordinator:
    """Advanced streaming coordinator for real-time frontend updates"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.event_history: Dict[str, List[StreamEvent]] = {}
        logger.info("Streaming Coordinator initialized")
    
    def create_session(self, query: str) -> str:
        """Create a new streaming session"""
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            'query': query,
            'start_time': time.time(),
            'current_agent': None,
            'progress': 0.0,
            'status': 'active'
        }
        self.event_history[session_id] = []
        return session_id
    
    def emit_event(self, session_id: str, event_type: StreamEventType, 
                   agent_name: str, data: Dict[str, Any]) -> StreamEvent:
        """Emit a streaming event"""
        step_id = str(uuid.uuid4())[:8]
        event = StreamEvent(
            event_type=event_type,
            agent_name=agent_name,
            timestamp=time.time(),
            data=data,
            session_id=session_id,
            step_id=step_id
        )
        
        # Store event in history
        if session_id in self.event_history:
            self.event_history[session_id].append(event)
        
        # Update session state
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['current_agent'] = agent_name
            
        return event
    
    def format_streaming_output(self, event: StreamEvent) -> str:
        """Format event for streaming output"""
        event_type = event.event_type
        agent_name = event.agent_name
        data = event.data
        
        # Format based on event type
        if event_type == StreamEventType.WORKFLOW_START:
            return f"🚀 **Starting Multi-Agent Workflow**\n📝 Query: {data.get('query', '')}\n\n"
        
        elif event_type == StreamEventType.AGENT_START:
            return f"🤖 **{agent_name.title()}** starting...\n"
        
        elif event_type == StreamEventType.AGENT_COMPLETE:
            duration = data.get('duration', 0)
            return f"✅ **{agent_name.title()}** completed ({duration:.1f}s)\n"
        
        elif event_type == StreamEventType.TOOL_CREATED:
            tool_name = data.get('tool_name', 'unknown')
            return f"🛠️ **Tool Created:** {tool_name}\n"
        
        elif event_type == StreamEventType.TOOL_EXECUTED:
            result_preview = data.get('result_preview', '')
            return f"⚡ **Tool Output:** {result_preview}\n"
        
        elif event_type == StreamEventType.WEB_SEARCH_RESULTS:
            count = data.get('result_count', 0)
            return f"🔍 **Found {count} web results**\n"
        
        elif event_type == StreamEventType.EVALUATION_UPDATE:
            completeness = data.get('completeness', 0)
            confidence = data.get('confidence', 0)
            return f"📊 **Answer Quality:** {completeness:.1%} complete, {confidence:.1%} confident\n"
        
        elif event_type == StreamEventType.SYNTHESIS_PROGRESS:
            return data.get('content', '')
        
        elif event_type == StreamEventType.WORKFLOW_COMPLETE:
            total_time = data.get('total_time', 0)
            final_confidence = data.get('final_confidence', 0)
            return f"\n🎉 **Workflow Complete** ({total_time:.1f}s, {final_confidence:.1%} confidence)\n"
        
        elif event_type == StreamEventType.ERROR:
            error_msg = data.get('error', 'Unknown error')
            return f"❌ **Error:** {error_msg}\n"
        
        return ""
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a streaming session"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        events = self.event_history.get(session_id, [])
        
        return {
            'session_id': session_id,
            'query': session['query'],
            'start_time': session['start_time'],
            'current_agent': session['current_agent'],
            'progress': session['progress'],
            'status': session['status'],
            'event_count': len(events),
            'last_event': events[-1].to_json() if events else None
        }
    
    def close_session(self, session_id: str) -> None:
        """Close a streaming session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['status'] = 'completed'
            logger.info(f"Session {session_id} closed")

class EnhancedLangGraphCoordinator:
    """
    Enhanced LangGraph coordinator with advanced streaming capabilities.
    
    This wraps around the base LangGraphCoordinator to provide:
    - Real-time progress updates
    - Detailed agent status tracking
    - Frontend-friendly event streaming
    - Session management
    """
    
    def __init__(self):
        from .langgraph_workflow import LangGraphCoordinator
        self.base_coordinator = LangGraphCoordinator()
        self.streaming_coordinator = StreamingCoordinator()
        logger.info("Enhanced LangGraph Coordinator with streaming initialized")
    
    def process_query_streaming(self, query: str) -> Generator[str, None, None]:
        """Process query with enhanced streaming capabilities"""
        
        # Create streaming session
        session_id = self.streaming_coordinator.create_session(query)
        
        # Emit workflow start event
        start_event = self.streaming_coordinator.emit_event(
            session_id, 
            StreamEventType.WORKFLOW_START,
            "coordinator",
            {"query": query}
        )
        yield self.streaming_coordinator.format_streaming_output(start_event)
        
        try:
            # Track agent execution
            current_agent = None
            agent_start_time = None
            
            # Process through base coordinator with enhanced event handling
            for chunk in self.base_coordinator.process_query_streaming(query):
                
                # Detect agent transitions
                if "**Coordinator Agent:**" in chunk:
                    current_agent = "coordinator"
                    agent_start_time = time.time()
                    event = self.streaming_coordinator.emit_event(
                        session_id, StreamEventType.AGENT_START, current_agent, {}
                    )
                    yield self.streaming_coordinator.format_streaming_output(event)
                
                elif "**Web Agent:**" in chunk:
                    if current_agent:
                        # Complete previous agent
                        duration = time.time() - agent_start_time if agent_start_time else 0
                        event = self.streaming_coordinator.emit_event(
                            session_id, StreamEventType.AGENT_COMPLETE, current_agent, 
                            {"duration": duration}
                        )
                        yield self.streaming_coordinator.format_streaming_output(event)
                    
                    current_agent = "web_agent"
                    agent_start_time = time.time()
                    event = self.streaming_coordinator.emit_event(
                        session_id, StreamEventType.AGENT_START, current_agent, {}
                    )
                    yield self.streaming_coordinator.format_streaming_output(event)
                
                elif "**MCP Agent:**" in chunk:
                    if current_agent:
                        # Complete previous agent
                        duration = time.time() - agent_start_time if agent_start_time else 0
                        event = self.streaming_coordinator.emit_event(
                            session_id, StreamEventType.AGENT_COMPLETE, current_agent, 
                            {"duration": duration}
                        )
                        yield self.streaming_coordinator.format_streaming_output(event)
                    
                    current_agent = "mcp_agent"
                    agent_start_time = time.time()
                    event = self.streaming_coordinator.emit_event(
                        session_id, StreamEventType.AGENT_START, current_agent, {}
                    )
                    yield self.streaming_coordinator.format_streaming_output(event)
                
                elif "**Evaluator Agent:**" in chunk:
                    if current_agent:
                        # Complete previous agent
                        duration = time.time() - agent_start_time if agent_start_time else 0
                        event = self.streaming_coordinator.emit_event(
                            session_id, StreamEventType.AGENT_COMPLETE, current_agent, 
                            {"duration": duration}
                        )
                        yield self.streaming_coordinator.format_streaming_output(event)
                    
                    current_agent = "evaluator"
                    agent_start_time = time.time()
                    event = self.streaming_coordinator.emit_event(
                        session_id, StreamEventType.AGENT_START, current_agent, {}
                    )
                    yield self.streaming_coordinator.format_streaming_output(event)
                
                elif "**Synthesizer Agent:**" in chunk:
                    if current_agent:
                        # Complete previous agent
                        duration = time.time() - agent_start_time if agent_start_time else 0
                        event = self.streaming_coordinator.emit_event(
                            session_id, StreamEventType.AGENT_COMPLETE, current_agent, 
                            {"duration": duration}
                        )
                        yield self.streaming_coordinator.format_streaming_output(event)
                    
                    current_agent = "synthesizer"
                    agent_start_time = time.time()
                    event = self.streaming_coordinator.emit_event(
                        session_id, StreamEventType.AGENT_START, current_agent, {}
                    )
                    yield self.streaming_coordinator.format_streaming_output(event)
                
                # Detect specific events
                elif "Tool Created:" in chunk:
                    tool_name = chunk.split("Tool Created:")[-1].strip()
                    event = self.streaming_coordinator.emit_event(
                        session_id, StreamEventType.TOOL_CREATED, current_agent or "unknown",
                        {"tool_name": tool_name}
                    )
                    yield self.streaming_coordinator.format_streaming_output(event)
                
                elif "Tool output:" in chunk:
                    output = chunk.split("Tool output:")[-1].strip()
                    event = self.streaming_coordinator.emit_event(
                        session_id, StreamEventType.TOOL_EXECUTED, current_agent or "unknown",
                        {"result_preview": output[:100]}
                    )
                    yield self.streaming_coordinator.format_streaming_output(event)
                
                elif "Found" in chunk and "web results" in chunk:
                    # Extract result count
                    import re
                    match = re.search(r'Found (\d+) web results', chunk)
                    if match:
                        count = int(match.group(1))
                        event = self.streaming_coordinator.emit_event(
                            session_id, StreamEventType.WEB_SEARCH_RESULTS, current_agent or "unknown",
                            {"result_count": count}
                        )
                        yield self.streaming_coordinator.format_streaming_output(event)
                
                elif "Completeness Score:" in chunk:
                    # Extract completeness score
                    import re
                    match = re.search(r'Completeness Score: ([\d.]+)', chunk)
                    if match:
                        score = float(match.group(1))
                        event = self.streaming_coordinator.emit_event(
                            session_id, StreamEventType.EVALUATION_UPDATE, current_agent or "unknown",
                            {"completeness": score, "confidence": score}
                        )
                        yield self.streaming_coordinator.format_streaming_output(event)
                
                # Stream the original chunk as synthesis progress if from synthesizer
                if current_agent == "synthesizer" and chunk.strip():
                    event = self.streaming_coordinator.emit_event(
                        session_id, StreamEventType.SYNTHESIS_PROGRESS, current_agent,
                        {"content": chunk}
                    )
                    yield self.streaming_coordinator.format_streaming_output(event)
                else:
                    # Stream original chunk
                    yield chunk
            
            # Complete final agent if any
            if current_agent and agent_start_time:
                duration = time.time() - agent_start_time
                event = self.streaming_coordinator.emit_event(
                    session_id, StreamEventType.AGENT_COMPLETE, current_agent, 
                    {"duration": duration}
                )
                yield self.streaming_coordinator.format_streaming_output(event)
            
            # Emit workflow completion
            session_info = self.streaming_coordinator.get_session_status(session_id)
            total_time = time.time() - session_info['start_time']
            
            complete_event = self.streaming_coordinator.emit_event(
                session_id, StreamEventType.WORKFLOW_COMPLETE, "coordinator",
                {"total_time": total_time, "final_confidence": 0.8}
            )
            yield self.streaming_coordinator.format_streaming_output(complete_event)
            
        except Exception as e:
            # Emit error event
            error_event = self.streaming_coordinator.emit_event(
                session_id, StreamEventType.ERROR, current_agent or "unknown",
                {"error": str(e)}
            )
            yield self.streaming_coordinator.format_streaming_output(error_event)
            
        finally:
            # Close session
            self.streaming_coordinator.close_session(session_id)
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active streaming sessions"""
        return [
            self.streaming_coordinator.get_session_status(session_id)
            for session_id in self.streaming_coordinator.active_sessions.keys()
        ] 