from src.agent.mcp import MCPAgent
from src.tool.mcp import MCPClients
import pytest

@pytest.fixture
def mcp_agent():
    agent = MCPAgent()
    return agent

def test_initialize_sse(mcp_agent):
    server_url = "http://example.com/mcp"
    asyncio.run(mcp_agent.initialize(connection_type="sse", server_url=server_url))
    assert mcp_agent.connection_type == "sse"
    assert mcp_agent.mcp_clients.sessions is not None

def test_initialize_stdio(mcp_agent):
    command = "python3 -m mcp_server"
    asyncio.run(mcp_agent.initialize(connection_type="stdio", command=command))
    assert mcp_agent.connection_type == "stdio"
    assert mcp_agent.mcp_clients.sessions is not None

def test_handle_special_tool(mcp_agent):
    result = ToolResult(base64_image="some_base64_string")
    asyncio.run(mcp_agent._handle_special_tool("test_tool", result))
    messages = mcp_agent.memory.get_messages()
    assert any("multimedia response" in msg.content for msg in messages)

def test_should_finish_execution(mcp_agent):
    assert mcp_agent._should_finish_execution("terminate") is True
    assert mcp_agent._should_finish_execution("continue") is False

def test_cleanup(mcp_agent):
    asyncio.run(mcp_agent.cleanup())
    assert mcp_agent.mcp_clients.sessions is None  # Assuming cleanup disconnects sessions

def test_refresh_tools(mcp_agent):
    asyncio.run(mcp_agent._refresh_tools())
    assert isinstance(mcp_agent.tool_schemas, dict)  # Ensure tool schemas are refreshed

def test_think_no_sessions(mcp_agent):
    mcp_agent.mcp_clients.sessions = None
    result = asyncio.run(mcp_agent.think())
    assert result is False
    assert mcp_agent.state == AgentState.FINISHED

def test_think_with_sessions(mcp_agent):
    mcp_agent.mcp_clients.sessions = [MCPClients()]
    mcp_agent.mcp_clients.tool_map = {"tool1": "description"}
    result = asyncio.run(mcp_agent.think())
    assert result is True  # Assuming think processes correctly with active sessions