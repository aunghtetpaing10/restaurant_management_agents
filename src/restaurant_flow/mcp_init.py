import os
from pathlib import Path
from typing import Dict, Any

from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "db" / "restaurant.db"

_MCP_ADAPTER: MCPServerAdapter | None = None
_MCP_TOOLS: Dict[str, Any] | None = None


def _resolve_db_path() -> str:
    """Resolve database path from env or fall back to project db file."""

    env_path = os.getenv("RESTAURANT_DB_PATH")
    if env_path:
        db_path = Path(env_path).expanduser()
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
    else:
        db_path = DEFAULT_DB_PATH

    return str(db_path)


def get_mcp_tools() -> Dict[str, Any]:
    """Return the cached MCP tools dictionary, creating the adapter if needed."""
    global _MCP_ADAPTER, _MCP_TOOLS

    if _MCP_TOOLS is None:
        db_path = _resolve_db_path()
        mcp_server_params = StdioServerParameters(
            command=os.getenv("RESTAURANT_MCP_COMMAND", "uvx"),
            args=[
                os.getenv("RESTAURANT_MCP_SERVER", "mcp-server-sqlite"),
                "--db-path",
                db_path,
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
