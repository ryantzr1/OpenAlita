import pytest
from src.agent.mcp import MCPAgent
from src.tool.base import ToolResult

@pytest.fixture
def mcp_agent():
    agent = MCPAgent()
    return agent

def test_initialize_sse(mcp_agent):
    server_url = "http://example.com/mcp"
    mcp_agent.connection_type = "sse"
    await mcp_agent.initialize(server_url=server_url)
    assert mcp_agent.mcp_clients.sessions is not None

def test_initialize_stdio(mcp_agent):
    command = "python3 -m mcp_server"
    mcp_agent.connection_type = "stdio"
    await mcp_agent.initialize(command=command)
    assert mcp_agent.mcp_clients.sessions is not None

def test_handle_special_tool_multimedia(mcp_agent):
    result = ToolResult(base64_image="some_base64_string")
    await mcp_agent._handle_special_tool("test_tool", result)
    assert "multimedia" in mcp_agent.memory.messages[-1].content

def test_should_finish_execution_terminate(mcp_agent):
    assert mcp_agent._should_finish_execution("terminate") is True

def test_should_finish_execution_non_terminate(mcp_agent):
    assert mcp_agent._should_finish_execution("some_tool") is False

async def test_cleanup(mcp_agent):
    await mcp_agent.cleanup()
    assert mcp_agent.mcp_clients.sessions is None