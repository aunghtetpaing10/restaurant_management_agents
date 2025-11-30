import ast
from typing import Type

from pydantic import BaseModel, Field

from crewai.tools import BaseTool


class MenuSearchInput(BaseModel):
    """Input schema for MenuSearchTool."""

    query: str = Field(
        ..., description="Guest question or keywords about menu items (e.g. 'vegan pasta')"
    )
    available_only: bool = Field(
        default=False,
        description="If true, only return menu items marked as currently available.",
    )


class MenuSearchTool(BaseTool):
    """Search restaurant menu items and return matching dishes."""

    name: str = "menu_search"
    description: str = (
        "Use this tool to answer guest questions about menu items. It queries the restaurant"
        " SQLite database (through the MCP SQLite server) and returns matching dishes with"
        " category, price, availability, and description."
    )
    args_schema: Type[BaseModel] = MenuSearchInput

    def _run(self, query: str, available_only: bool = False) -> str:
        # Import here to avoid circular dependency and ensure MCP is initialized
        from restaurant_flow.mcp_init import get_mcp_tools
        
        try:
            tools = get_mcp_tools()
        except Exception as e:
            return f"MenuSearchTool error: Failed to initialize MCP tools - {str(e)}"
        
        # ToolCollection is iterable, find the read_query tool
        read_query_tool = None
        try:
            for tool in tools:
                if hasattr(tool, 'name') and tool.name == "read_query":
                    read_query_tool = tool
                    break
        except Exception as e:
            return f"MenuSearchTool error: Failed to find read_query tool - {str(e)}"

        if read_query_tool is None:
            return "MenuSearchTool error: read_query tool is not available from MCP server."

        cleaned_query = query.replace("'", "''")

        sql = (
            "SELECT name, category, price, description, is_available "
            "FROM menu_items "
            f"WHERE (name LIKE '%{cleaned_query}%' OR description LIKE '%{cleaned_query}%')"
        )

        if available_only:
            sql += " AND is_available = 1"

        sql += " ORDER BY category, name LIMIT 10"

        try:
            # CrewAIMCPTool uses .run() method, not .invoke()
            response = read_query_tool.run(query=sql)
        except Exception as e:
            return f"MenuSearchTool error: Failed to execute query - {str(e)}"

        # The MCP read_query tool returns a string representation of a Python list
        # Parse it to get the actual data
        try:
            if isinstance(response, str):
                # Parse Python literal (list of dicts with single quotes)
                results = ast.literal_eval(response)
            elif isinstance(response, list):
                results = response
            elif isinstance(response, dict):
                if not response.get("success", True):
                    return f"MenuSearchTool error: {response.get('error', 'Unknown error')}"
                results = response.get("results", response.get("data", []))
            else:
                return f"MenuSearchTool received unexpected response format: {type(response).__name__}"
        except (ValueError, SyntaxError) as e:
            return f"MenuSearchTool error: Failed to parse response - {str(e)}"
        if not results:
            return "No matching menu items found for that query."

        lines = []
        for item in results:
            availability = "Available" if item.get("is_available") else "Unavailable"
            price_value = item.get("price")
            if price_value is None:
                price_text = "Price unavailable"
            else:
                price_text = f"${float(price_value):.2f}"

            line = (
                f"{item.get('name', 'Unknown')} ({item.get('category', 'Unknown')}): "
                f"{price_text} - {availability}. "
                f"{item.get('description', 'No description provided.')}"
            )
            lines.append(line)

        return "\n".join(lines)
