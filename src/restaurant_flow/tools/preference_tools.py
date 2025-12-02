import ast
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class CustomerPreferenceInput(BaseModel):
    """Input schema for CustomerPreferenceTool."""
    
    action: str = Field(
        ..., description="Action to perform: 'get', 'set', or 'get_all'"
    )
    customer_id: int | None = Field(
        default=None, description="Customer ID (required for all actions)"
    )
    preference_key: str | None = Field(
        default=None, description="Preference key (required for 'get' and 'set')"
    )
    preference_value: str | None = Field(
        default=None, description="Preference value (required for 'set')"
    )


class CustomerPreferenceTool(BaseTool):
    """Manage customer preferences in database."""
    
    name: str = "customer_preference"
    description: str = "Get or set customer preferences. Actions: get (retrieve one), set (save/update), get_all (retrieve all for customer)."
    args_schema: Type[BaseModel] = CustomerPreferenceInput
    
    def _run(self, action: str, customer_id: int | None = None, preference_key: str | None = None, preference_value: str | None = None) -> str:
        from restaurant_flow.mcp_init import get_mcp_tools
        
        mcp_tools = get_mcp_tools()
        read_query_tool = mcp_tools["read_query"]
        write_query_tool = mcp_tools["write_query"]
        
        if not customer_id:
            return "Error: customer_id is required"
        
        if action == "get":
            if not preference_key:
                return "Error: preference_key is required for 'get' action"
            
            query = f"""
                SELECT preference_value, updated_at 
                FROM customer_preferences 
                WHERE customer_id = {customer_id} AND preference_key = '{preference_key}'
            """
            
            try:
                response = read_query_tool.run(query=query)
                results = ast.literal_eval(response)
                
                if not results:
                    return f"No preference found for key '{preference_key}'"
                
                pref = results[0]
                return f"{preference_key}: {pref.get('preference_value')}"
            except Exception as e:
                return f"Error retrieving preference: {str(e)}"
        
        elif action == "set":
            if not preference_key or not preference_value:
                return "Error: preference_key and preference_value are required for 'set' action"
            
            # Use INSERT OR REPLACE to handle updates
            query = f"""
                INSERT INTO customer_preferences (customer_id, preference_key, preference_value, created_at, updated_at)
                VALUES ({customer_id}, '{preference_key}', '{preference_value}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(customer_id, preference_key) 
                DO UPDATE SET preference_value = '{preference_value}', updated_at = CURRENT_TIMESTAMP
            """
            
            try:
                write_query_tool.run(query=query)
                return f"Saved preference: {preference_key} = {preference_value}"
            except Exception as e:
                return f"Error saving preference: {str(e)}"
        
        elif action == "get_all":
            query = f"""
                SELECT preference_key, preference_value 
                FROM customer_preferences 
                WHERE customer_id = {customer_id}
                ORDER BY preference_key
            """
            
            try:
                response = read_query_tool.run(query=query)
                results = ast.literal_eval(response)
                
                if not results:
                    return f"No preferences found for customer {customer_id}"
                
                output = f"Preferences for customer {customer_id}:\n"
                for pref in results:
                    output += f"  {pref.get('preference_key')}: {pref.get('preference_value')}\n"
                
                return output.strip()
            except Exception as e:
                return f"Error retrieving preferences: {str(e)}"
        
        else:
            return f"Error: Unknown action '{action}'. Use 'get', 'set', or 'get_all'."
