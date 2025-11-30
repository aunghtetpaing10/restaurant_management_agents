from typing import List
from crewai import Agent
from crewai.flow.flow import Flow, listen, or_, start, router
from pydantic import BaseModel, Field

from restaurant_flow.tools.custom_tool import MenuSearchTool, OrderLookupTool


class MenuResponse(BaseModel):
    """Structured output for menu inquiries."""

    menu_items: List[str] = Field(
        description="List of menu item names found (e.g., ['Korean BBQ Chicken Wings', 'Caesar Salad'])"
    )
    prices: List[float] = Field(
        description="Prices for each menu item in order", default_factory=list
    )


class OrderResponse(BaseModel):
    """Structured output for order inquiries."""

    order_id: int = Field(description="ID of the order")
    items_ordered: List[str] = Field(
        description="List of ordered items (e.g., ['Caesar Salad', 'Margherita Pizza'])"
    )
    total_amount: float = Field(description="Total amount of the order")
    order_status: str = Field(description="Status of the order")


class FinalResponse(BaseModel):
    """Structured output for the final composed response."""

    customer_message_summary: str = Field(
        description="Brief summary of what the customer asked"
    )
    final_response: str = Field(
        description="The polished, professional final response to send to the customer"
    )


class RestaurantState(BaseModel):
    customer_message: str = "What is the order status of ID #3"
    menu_response: MenuResponse | None = None
    order_response: OrderResponse | None = None
    final_response: FinalResponse | None = None


class RestaurantFlow(Flow[RestaurantState]):
    @start()
    def receive_message(self):
        print("Starting the structured flow")
        self.state.customer_message = self.state.customer_message

    @router(receive_message)
    def classify_intent(self):
        print("[CLASSIFY] Classifying intent...")
        message = self.state.customer_message.lower()

        # Check for order-related keywords
        if any(
            keyword in message for keyword in ["order", "status", "check order", "#"]
        ):
            print("[CLASSIFY] Intent: ORDER")
            return "order"
        # Check for menu-related keywords
        elif any(
            keyword in message
            for keyword in ["menu", "appetizers", "dish", "food", "what do you have"]
        ):
            print("[CLASSIFY] Intent: MENU")
            return "menu"
        else:
            # Default to menu
            print("[CLASSIFY] Intent: MENU (default)")
            return "menu"

    @listen("menu")
    def handle_menu(self):
        print("[MENU] Processing menu inquiry...")

        menu_specialist = Agent(
            role="Menu Specialist",
            goal="Provide detailed, accurate menu information",
            backstory=(
                "You are a knowledgeable menu expert who knows every dish, ingredient, "
                "and dietary option. You use the menu_search tool to query the database."
            ),
            tools=[MenuSearchTool()],
            verbose=True,
        )

        query = f"""
        Customer inquiry: {self.state.customer_message}
        
        Your task:
        1. Use the menu_search tool to find relevant menu items or categories from the database
        2. Provide comprehensive information about:
           - Dish names
           - Prices
        3. Structure your response according to specified structure.
        
        """

        result = menu_specialist.kickoff(query, response_format=MenuResponse)

        print("\n" + "=" * 60)
        print("[DEBUG] Menu Specialist Result:")
        print(f"  - Items: {result.pydantic.menu_items}")
        print(f"  - Prices: {result.pydantic.prices}")
        print("=" * 60)

        # Update state directly instead of returning
        self.state.menu_response = result.pydantic
        return self.state.menu_response

    @listen("order")
    def handle_order(self):
        print("[ORDER] Processing order request...")

        # Pre-extract order ID from message to make it explicit
        import re

        message = self.state.customer_message
        order_id_match = re.search(
            r"#(\d+)|ID[:\s]+(\d+)|order[:\s]+(\d+)", message, re.IGNORECASE
        )
        extracted_id = None
        if order_id_match:
            extracted_id = next((g for g in order_id_match.groups() if g), None)
            print(f"[DEBUG] Extracted Order ID from message: {extracted_id}")

        order_handler = Agent(
            role="Order Handler",
            goal="Process orders and provide order status information",
            backstory=(
                "You are an efficient order processing specialist who handles order creation, "
                "status checks, modifications, and cancellations. You use the order_lookup tool "
                "to interact with the order management system."
            ),
            tools=[OrderLookupTool()],
            verbose=True,
        )

        # Build query with explicit order ID if found
        if extracted_id:
            query = f"""
        Customer is asking about ORDER ID: {extracted_id}
        
        CRITICAL INSTRUCTION: You MUST use order_id={extracted_id} when calling the order_lookup tool.
        DO NOT use any other number. The customer specifically asked about order #{extracted_id}.
        
        Step 1: Call the order_lookup tool with these EXACT parameters:
        - action: 'lookup_by_id'
        - order_id: {extracted_id}  <-- USE THIS EXACT NUMBER
        
        Step 2: Take the response from the tool and fill in the OrderResponse:
        - order_id: {extracted_id}  <-- USE THIS EXACT NUMBER
        - items_ordered: [list from tool response]
        - total_amount: [amount from tool response]
        - order_status: [status from tool response]
        
        If the order is not found, still use order_id={extracted_id} in your response with status "Not Found".
        
        DO NOT MAKE UP ORDER IDS. USE {extracted_id} ONLY.
        """
        else:
            query = f"""
        Customer request: '{self.state.customer_message}'
        
        Your task:
        1. FIRST, carefully extract the order ID from the customer's message:
           - Look for patterns like "ID #18", "order #18", "#18", "order 18", "ID: 18"
           - Extract ONLY the numeric part
        
        2. Use the order_lookup tool with the EXACT order ID you extracted:
           - action: 'lookup_by_id'
           - order_id: [the number you extracted]
        
        3. Provide comprehensive information from the database response.
        
        4. Structure your response according to the specified OrderResponse format.
        """

        result = order_handler.kickoff(query, response_format=OrderResponse)

        print("\n" + "=" * 60)
        print("[DEBUG] Order Handler Result:")
        print(f"  - Order ID: {result.pydantic.order_id}")
        print(f"  - Items: {result.pydantic.items_ordered}")
        print(f"  - Total: ${result.pydantic.total_amount}")
        print(f"  - Status: {result.pydantic.order_status}")
        print("=" * 60)

        # Update state directly instead of returning
        self.state.order_response = result.pydantic
        return self.state.order_response

    @listen(or_(handle_menu, handle_order))
    def deliver_response(self):
        composer = Agent(
            role="Response Composer",
            goal="Create polished, professional customer responses",
            backstory=(
                "You are an expert at crafting warm, professional customer service responses. "
                "You take information from specialists and compose clear, friendly, helpful messages "
                "that address the customer's needs while maintaining a professional tone."
            ),
            verbose=True,
        )

        # Format the specialist response data for the composer
        specialist_data = ""
        if self.state.menu_response:
            menu = self.state.menu_response
            specialist_data = f"""
            Menu Specialist Response:
            - Items found: {", ".join(menu.menu_items)}
            - Prices: {menu.prices}
            """
        elif self.state.order_response:
            order = self.state.order_response
            specialist_data = f"""
            Order Handler Response:
            - Order ID: {order.order_id}
            - Items ordered: {", ".join(order.items_ordered)}
            - Total amount: ${order.total_amount}
            - Order status: {order.order_status}
            """

        print("\n" + "=" * 60)
        print("[DEBUG] Data being sent to Response Composer:")
        print(specialist_data)
        print("=" * 60)

        query = f"""
        Customer inquiry: {self.state.customer_message}
        
        {specialist_data}
        
        Your task:
        1. Compose a final, polished customer response that:
           - Addresses the customer's request directly and completely
           - Incorporates all key information from the specialist
           - Uses appropriate tone based on customer sentiment
           - Is clear, concise, and easy to understand
           - Includes a warm, friendly closing
           - Adds any helpful next steps or suggestions
        2. Provide structured output according to specified structure.
        """

        result = composer.kickoff(query, response_format=FinalResponse)

        print("\n" + "=" * 60)
        print("[DELIVER] Final response to customer:")
        print("=" * 60)
        print(result.pydantic.final_response)
        print("=" * 60)

        self.state.final_response = result.pydantic
        return self.state.final_response


def kickoff():
    """Run the restaurant flow."""
    restaurant_flow = RestaurantFlow()
    restaurant_flow.kickoff()


def plot():
    restaurant_flow = RestaurantFlow()
    restaurant_flow.plot()


if __name__ == "__main__":
    kickoff()
