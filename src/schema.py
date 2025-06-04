from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class Message(BaseModel):
    """Represents a message in the system."""
    role: str
    content: str
    timestamp: Optional[str] = Field(default=None)

class AgentState(str):
    """Represents the state of the agent."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"

class ToolSchema(BaseModel):
    """Defines the schema for a tool."""
    name: str
    inputSchema: Dict[str, Any]
    outputSchema: Dict[str, Any]

class ToolResult(BaseModel):
    """Represents the result of a tool execution."""
    success: bool
    output: Any
    base64_image: Optional[str] = Field(default=None)