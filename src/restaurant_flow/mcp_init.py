import os
from typing import Dict, Any

from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter

_MCP_ADAPTER: MCPServerAdapter | None = None
_MCP_TOOLS: Dict[str, Any] | None = None


def get_mcp_tools() -> Dict[str, Any]:
    """Return the cached MCP tools dictionary, creating the adapter if needed."""
    global _MCP_ADAPTER, _MCP_TOOLS

    if _MCP_TOOLS is None:
        mcp_server_params = StdioServerParameters(
            command="uvx",
            args=[
                "mcp-server-sqlite",
                "--db-path",
                "d:/AI Agents Projects/restaurant_management_agents/restaurant_flow/db/restaurant.db",
            ],
            env={**os.environ},
        )
        _MCP_ADAPTER = MCPServerAdapter(mcp_server_params)
        _MCP_TOOLS = _MCP_ADAPTER.__enter__()

    return _MCP_TOOLS


def close_mcp_tools() -> None:
    """Close the MCP adapter if it was initialized."""
    global _MCP_ADAPTER, _MCP_TOOLS

    if _MCP_ADAPTER is not None:
        _MCP_ADAPTER.__exit__(None, None, None)
        _MCP_ADAPTER = None
        _MCP_TOOLS = None
