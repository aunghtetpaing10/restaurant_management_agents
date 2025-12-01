from typing import List
from crewai import LLM, Agent
from crewai.flow.flow import Flow, listen, or_, start, router
from pydantic import BaseModel, Field

from restaurant_flow.tools.custom_tool import (
    MenuSearchTool,
    OrderLookupTool,
    ReservationLookupTool,
    CustomerLookupTool,
)

llm = LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")


class IntentClassification(BaseModel):
    """Intent classifier structured output."""

    intent: str = Field(
        description="Detected intent category (menu_inquiry, order_request, reservation_request, general_question, complaint, unclear, other)"
    )
    requires_escalation: bool = Field(
        default=False,
        description="True if the message contains a complaint, urgent issue, or requires human intervention",
    )
    confidence: str = Field(
        default="high",
        description="Confidence level in the classification: high, medium, or low",
    )


class MenuResponse(BaseModel):
    """Structured output for menu inquiries."""

    menu_items: List[str] = Field(
        description="List of menu item names found (e.g., ['Korean BBQ Chicken Wings', 'Caesar Salad'])"
    )
    prices: List[float] = Field(
        description="Prices for each menu item in order", default_factory=list
    )


class OrderItemDetail(BaseModel):
    """Structured summary for each ordered item."""

    menu: str = Field(description="Menu item name (e.g., 'Caesar Salad')")
    price: str = Field(description="Price string including currency (e.g., '$12.99')")
    quantity: int = Field(description="Quantity ordered")


class OrderResponse(BaseModel):
    """Structured output for order inquiries."""

    order_id: int | None = Field(
        default=None,
        description="ID of the order (if known or after creation)",
    )
    items_ordered: List[OrderItemDetail] = Field(
        default_factory=list,
        description="List of ordered items with menu name, price string, and quantity",
    )
    total_amount: float = Field(description="Total amount of the order")
    order_status: str = Field(description="Status of the order")


class ReservationResponse(BaseModel):
    """Structured output for reservation inquiries."""

    reservation_id: int | None = Field(
        default=None, description="Reservation ID if available"
    )
    party_size: int = Field(description="Number of guests")
    reservation_datetime: str = Field(description="Reservation date (YYYY-MM-DD)")
    status: str = Field(
        description="Status of reservation (confirmed, waitlisted, cancelled, etc.)"
    )
    special_requests: str = Field(
        default="",
        description="Any special requests from the customer",
    )


class FinalResponse(BaseModel):
    """Structured output for the final composed response."""

    customer_message_summary: str = Field(
        description="Brief summary of what the customer asked"
    )
    final_response: str = Field(
        description="The polished, professional final response to send to the customer"
    )


class RestaurantState(BaseModel):
    customer_message: str = (
        "I want to see my order history for Liam Patel"
    )
    classification: IntentClassification | None = None
    menu_response: MenuResponse | None = None
    order_response: OrderResponse | None = None
    reservation_response: ReservationResponse | None = None
    final_response: FinalResponse | None = None


class RestaurantFlow(Flow[RestaurantState]):
    @start()
    def receive_message(self):
        print("Starting the structured flow")
        self.state.customer_message = self.state.customer_message

    @router(receive_message)
    def classify_intent(self):
        print("[CLASSIFY] Classifying intent with agent...")

        classifier = Agent(
            role="Intent Classifier",
            goal="Determine customer intent",
            backstory="Restaurant triage specialist who routes customers to the right service.",
            verbose=True,
            llm=llm,
        )

        query = f"""
        Message: "{self.state.customer_message}"

        Classify intent (distinguish ASKING vs DOING):
        - menu_inquiry: wants menu info/prices
        - order_request: ACTIVELY ordering items OR checking order by ID
        - reservation_request: ACTIVELY booking with date/time/size OR checking reservation
        - general_question: asking HOW to do something or about policies (NOT doing it)
        - complaint: dissatisfaction/problem
        - unclear: ambiguous
        - other: doesn't fit above

        Set requires_escalation=true if: complaints, anger, manager request, threats, serious issues, hostile tone.
        Set confidence: high/medium/low based on clarity.
        """

        result = classifier.kickoff(query, response_format=IntentClassification)
        classification = result.pydantic

        print(f"[CLASSIFY] Intent: {classification.intent} | Escalation: {classification.requires_escalation} | Confidence: {classification.confidence}")

        self.state.classification = classification

        # Check for escalation first
        if classification.requires_escalation:
            print("[CLASSIFY] ESCALATION REQUIRED - Routing to escalation handler")
            return "escalation"

        intent = classification.intent.lower().strip()
        if intent in {"menu", "menu_inquiry"}:
            print("[CLASSIFY] Routing to menu specialist")
            return "menu_inquiry"
        if intent in {"order", "order_request"}:
            print("[CLASSIFY] Routing to order handler")
            return "order_request"
        if intent in {"reservation", "reservation_request"}:
            print("[CLASSIFY] Routing to reservation agent")
            return "reservation_request"
        if intent in {"complaint"}:
            print("[CLASSIFY] Complaint detected - Routing to escalation handler")
            return "escalation"
        if intent in {"unclear"}:
            print("[CLASSIFY] Unclear intent - Routing to fallback handler")
            return "fallback"
        if intent in {"general_question", "other"}:
            print("[CLASSIFY] General/other intent - Routing to fallback handler")
            return "fallback"

        # Final safety fallback
        print("[CLASSIFY] Unrecognized intent - Routing to fallback handler")
        return "fallback"

    @listen("menu_inquiry")
    def handle_menu(self):
        print("[MENU] Processing menu inquiry...")

        menu_specialist = Agent(
            role="Menu Specialist",
            goal="Provide menu information",
            backstory="Menu expert who uses menu_search tool.",
            tools=[MenuSearchTool()],
            verbose=True,
            llm=llm,
        )

        query = f"""
        Customer: {self.state.customer_message}
        
        Task: Use menu_search tool with customer's query to find matching menu items.
        Return: dish names and prices in MenuResponse format.
        """

        result = menu_specialist.kickoff(query, response_format=MenuResponse)

        print(f"[MENU] Found {len(result.pydantic.menu_items)} items")

        # Update state directly instead of returning
        self.state.menu_response = result.pydantic
        return self.state.menu_response

    @listen("order_request")
    def handle_order(self):
        print(f"[ORDER] Processing order request: '{self.state.customer_message}'")

        order_handler = Agent(
            role="Order Handler",
            goal="Process orders and status",
            backstory="Order specialist using order_lookup, menu_search, and customer_lookup tools.",
            tools=[OrderLookupTool(), MenuSearchTool(), CustomerLookupTool()],
            verbose=True,
            llm=llm,
        )

        query = f"""
        CUSTOMER MESSAGE: '{self.state.customer_message}'

        CRITICAL: Read the customer message above carefully. Extract ONLY the items mentioned in THIS specific message.
        
        For NEW order:
        1. Extract EXACT item names from customer message (e.g., "Spicy Thai Basil Stir Fry", "Warm Chocolate Lava Cake")
        2. Extract quantities for each item
        3. Extract customer_id
        4. For EACH item the customer mentioned, use menu_search tool to find it in the database
           Example: menu_search(query="Spicy Thai Basil Stir Fry")
        5. From menu_search results, extract the ID for each item
           The tool returns: "ID: 28 | Spicy Thai Basil Stir Fry (Entrees): $17.00 - Available..."
           Extract the number after "ID:" - this is the menu_item_id you need
        6. Once you have all menu_item_ids from menu_search, call order_lookup:
           action='create', customer_id=X, items='[{{"menu_item_id": Y, "quantity": Z}}]'
        7. Do NOT guess or make up menu IDs. ALWAYS use menu_search first to get the correct IDs.
        
        For EXISTING order lookup:
        1. Extract identifier from message (order ID, phone, or name)
        2. If order ID provided (e.g., "#18", "order 42"): call order_lookup(action='lookup_by_id', order_id=18)
        3. If phone OR name provided:
           a. If phone given (e.g., "phone is 555-1005"): call order_lookup(action='lookup_by_phone', phone="555-1005")
           b. If name given (e.g., "name is Mia Johnson"): 
              - First use customer_lookup(query="Mia Johnson") to get phone
              - Then use order_lookup(action='lookup_by_phone', phone=<found_phone>)
           - Tool output example:
             "Orders for Mia Johnson (phone: 555-1005):\n\nOrder #27 - in_progress\n  Date: 2025-12-01 18:30:00\n  Total: $45.00"
           - Use the most recent order entry to populate OrderResponse (order_id, items if provided, total_amount, order_status)
        
        After tool call: If JSON_SUMMARY present, parse it. Otherwise parse text output to fill OrderResponse fields. Ensure reported items/status match customer request.
        """ 

        result = order_handler.kickoff(query, response_format=OrderResponse)

        print(f"[ORDER] ID: {result.pydantic.order_id} | Items: {len(result.pydantic.items_ordered)} | Total: ${result.pydantic.total_amount} | Status: {result.pydantic.order_status}")

        # Update state directly instead of returning
        self.state.order_response = result.pydantic
        return self.state.order_response

    @listen("reservation_request")
    def handle_reservation(self):
        print(f"[RESERVATION] Processing reservation request: '{self.state.customer_message}'")

        reservation_agent = Agent(
            role="Reservation Agent",
            goal="Manage reservations",
            backstory="Reservation specialist using reservation_lookup and customer_lookup tools.",
            tools=[ReservationLookupTool(), CustomerLookupTool()],
            verbose=True,
            llm=llm,
        )

        query = f"""
        CUSTOMER MESSAGE: '{self.state.customer_message}'

        CRITICAL: Read the customer message carefully. Use ONLY the information provided in the message.
        
        For NEW reservation:
        1. Extract from message: party_size, date/time, customer identifier
           - Convert relative dates: "tomorrow" = 2025-12-03, "today" = 2025-12-02, etc.
           - Convert times: "7pm" = "19:00", "2:30pm" = "14:30", etc.
        
        2. Get customer_id:
           - If customer provided name (e.g., "name is Liam Patel"):
             Call customer_lookup(query="Liam Patel") to find their ID
           - If customer provided phone (e.g., "phone is 555-1005"):
             Call customer_lookup(query="555-1005") to find their ID
           - If customer provided email:
             Call customer_lookup(query="email@example.com") to find their ID
           - If customer provided customer_id directly: use it
           - DO NOT make up information. Use ONLY what customer provided.
           - Tool returns: "ID: 4 | Liam Patel | Phone: 555-1004 | Email: liam.patel@example.com"
           - Extract the number after "ID:" - this is the customer_id
        
        3. Create reservation:
           Once you have customer_id, party_size, date, and time, call:
           reservation_lookup(action='create', customer_id=X, party_size=Y, reservation_date='YYYY-MM-DD', reservation_time='HH:MM')
        
        4. If ANY critical data is missing, set status="awaiting_details" and explain what's needed. Do NOT call tools.
        
        For EXISTING reservation lookup:
        1. Extract reservation identifier (reservation_id, phone, or name) from message
        2. If reservation_id provided: call reservation_lookup(action='lookup_by_id', reservation_id=X)
        3. If phone OR name provided:
           a. If phone given (e.g., "phone is 555-1005"): call reservation_lookup(action='lookup_by_phone', phone="555-1005")
           b. If name given (e.g., "name is Mia Johnson"):
              - First use customer_lookup(query="Mia Johnson") to get phone
              - Tool returns: "ID: 5 | Mia Johnson | Phone: 555-1005 | Email: mia.johnson@example.com"
              - Extract phone number from result
              - Then call reservation_lookup(action='lookup_by_phone', phone="555-1005")
           - Tool output looks like:
             "Reservations for Mia Johnson (phone: 555-1005):\n\nReservation #15 - confirmed\n  Date & Time: 2025-12-04 19:00:00\n  Party Size: 6 people"
           - Parse the first reservation entry (closest upcoming) to fill ReservationResponse:
             reservation_id=15, party_size=6, reservation_datetime="2025-12-04 19:00:00", status="confirmed"
        
        Fill ReservationResponse fields with the parsed data. If no reservations returned, set status="awaiting_details" and explain that no reservation exists.
        """

        result = reservation_agent.kickoff(query, response_format=ReservationResponse)

        print(f"[RESERVATION] ID: {result.pydantic.reservation_id} | Party: {result.pydantic.party_size} | Time: {result.pydantic.reservation_datetime} | Status: {result.pydantic.status}")

        self.state.reservation_response = result.pydantic
        return self.state.reservation_response

    @listen("escalation")
    def handle_escalation(self):
        print("[ESCALATION] Processing escalation request...")

        escalation_agent = Agent(
            role="Customer Service Manager",
            goal="Handle escalations with empathy",
            backstory="Senior manager who de-escalates and resolves complaints.",
            verbose=True,
            llm=llm,
        )

        query = f"""
        Message: '{self.state.customer_message}' (flagged: {self.state.classification.intent})
        
        Acknowledge concern with empathy. Apologize if needed. Explain immediate action. Offer solution/compensation. Provide contact info. De-escalate professionally.
        """

        result = escalation_agent.kickoff(query, response_format=FinalResponse)

        print("[ESCALATION] Response generated")

        self.state.final_response = result.pydantic
        return self.state.final_response

    @listen("fallback")
    def handle_fallback(self):
        print("[FALLBACK] Processing unclear or general request...")

        fallback_agent = Agent(
            role="General Support Agent",
            goal="Assist with general inquiries",
            backstory="Friendly support agent who clarifies requests and guides customers.",
            verbose=True,
            llm=llm,
        )

        query = f"""
        Message: '{self.state.customer_message}' ({self.state.classification.intent}, {self.state.classification.confidence})
        
        If unclear: ask clarifying questions. If general: provide info or guide to right service. Offer options (menu/order/reservation). Be friendly and helpful.
        """

        result = fallback_agent.kickoff(query, response_format=FinalResponse)

        print("[FALLBACK] Response generated")

        self.state.final_response = result.pydantic
        return self.state.final_response

    @listen(or_(handle_menu, handle_order, handle_reservation))
    def deliver_response(self):
        composer = Agent(
            role="Response Composer",
            goal="Create polished responses",
            backstory="Expert at crafting professional customer service messages.",
            verbose=True,
            llm=llm,
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
            order_items_lines = "\n".join(
                [
                    f"                • {item.menu} x{item.quantity} @ {item.price}"
                    for item in order.items_ordered
                ]
            )
            if not order_items_lines:
                order_items_lines = "                • (none)"

            specialist_data = f"""
            Order Handler Response:
            - Order ID: {order.order_id}
            - Items:
{order_items_lines}
            - Total amount: ${order.total_amount}
            - Order status: {order.order_status}
            """
        elif self.state.reservation_response:
            reservation = self.state.reservation_response
            specialist_data = f"""
            Reservation Agent Response:
            - Reservation ID: {reservation.reservation_id}
            - Party size: {reservation.party_size}
            - Date: {reservation.reservation_datetime}
            - Status: {reservation.status}
            - Special requests: {reservation.special_requests}
            """

        query = f"""
        Customer: {self.state.customer_message}
        
        Specialist Data:
        {specialist_data}
        
        Task: Compose final customer response.
        - Use ALL information from specialist data above
        - Address customer's request directly
        - Professional and friendly tone
        - Include helpful next steps
        - Return in FinalResponse format
        """

        result = composer.kickoff(query, response_format=FinalResponse)

        print("[DELIVER] Final response composed")

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
