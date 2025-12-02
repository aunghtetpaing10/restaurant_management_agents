from typing import List
import time
from crewai import LLM, Agent
from crewai.flow.flow import Flow, listen, or_, start, router
from pydantic import BaseModel, Field

from restaurant_flow.tools.custom_tool import (
    MenuSearchTool,
    OrderLookupTool,
    ReservationLookupTool,
    CustomerLookupTool,
)
from restaurant_flow.tools.preference_tools import CustomerPreferenceTool


llm = LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")


def retry_agent_call(agent_func, max_retries=2, delay=1):
    """Simple retry wrapper for agent calls with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return agent_func()
        except Exception as e:
            if attempt == max_retries:
                print(
                    f"[ERROR] Agent call failed after {max_retries + 1} attempts: {str(e)}"
                )
                raise
            print(
                f"[RETRY] Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}"
            )
            time.sleep(delay)
            delay *= 2


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
    
    @classmethod
    def model_validate(cls, obj):
        """Custom validation to handle string-encoded lists."""
        if isinstance(obj, dict) and isinstance(obj.get('items_ordered'), str):
            import json
            try:
                obj['items_ordered'] = json.loads(obj['items_ordered'])
            except (json.JSONDecodeError, TypeError):
                obj['items_ordered'] = []
        return super().model_validate(obj)


class ReservationResponse(BaseModel):
    """Structured output for reservation inquiries."""

    reservation_id: int | None = Field(
        default=None, description="Reservation ID if available"
    )
    party_size: int | None = Field(
        default=None, description="Number of guests (if known)"
    )
    reservation_datetime: str | None = Field(
        default=None, description="Reservation date/time (if known)"
    )
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
    customer_message: str = "Can you make it 2 for my last order for Noah Chen"
    classification: IntentClassification | None = None
    menu_response: MenuResponse | None = None
    order_response: OrderResponse | None = None
    reservation_response: ReservationResponse | None = None
    final_response: FinalResponse | None = None
    current_customer_id: int | None = Field(
        default=None, description="Database ID of current customer for memory tracking"
    )


class RestaurantFlow(Flow[RestaurantState]):
    def _update_memory(self, key: str, value: str):
        """Save customer preference to database."""
        if not self.state.current_customer_id:
            print("[MEMORY] No customer_id set, skipping memory save")
            return

        pref_tool = CustomerPreferenceTool()
        result = pref_tool._run(
            action="set",
            customer_id=self.state.current_customer_id,
            preference_key=key,
            preference_value=value,
        )
        print(f"[MEMORY] {result}")

    def _get_context_summary(self) -> str:
        """Get customer preferences from database."""
        if not self.state.current_customer_id:
            return "No previous context."

        # Get preferences from database
        pref_tool = CustomerPreferenceTool()
        prefs_result = pref_tool._run(
            action="get_all", customer_id=self.state.current_customer_id
        )

        if prefs_result.startswith("No preferences"):
            return "No previous context."

        return prefs_result

    @start()
    def receive_message(self):
        print("Starting the structured flow")
        
        # Extract customer_id early if customer name is mentioned
        if not self.state.current_customer_id and "for" in self.state.customer_message.lower():
            import re
            name_match = re.search(
                r"for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                self.state.customer_message,
            )
            if name_match:
                customer_name = name_match.group(1)
                lookup_tool = CustomerLookupTool()
                lookup_result = lookup_tool._run(query=customer_name)
                id_match = re.search(r"ID:\s+(\d+)", lookup_result)
                if id_match:
                    self.state.current_customer_id = int(id_match.group(1))
                    print(f"[MEMORY] Identified customer_id: {self.state.current_customer_id} ({customer_name})")
        
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

        context = self._get_context_summary()

        query = f"""
        Current Message: "{self.state.customer_message}"
        
        Context from previous conversation:
        {context}

        Classify intent (distinguish ASKING vs DOING):
        - menu_inquiry: wants menu info/prices
        - order_request: ACTIVELY ordering items OR checking/modifying existing order
          Examples: "I want pizza", "order #42", "make it 2", "change my last order", "add fries"
        - reservation_request: ACTIVELY booking with date/time/size OR checking/modifying reservation
          Examples: "book table for 4", "reservation #15", "change to 8pm"
        - general_question: asking HOW to do something or about policies (NOT doing it)
        - complaint: dissatisfaction/problem
        - unclear: ambiguous
        - other: doesn't fit above

        IMPORTANT: If context shows "last_order_id" or "last_reservation_id":
        - "make it 2", "change to X", "modify" → classify as order_request or reservation_request
        - Use context to understand what "it" or "my last order" refers to
        
        Set requires_escalation=true if: complaints, anger, manager request, threats, serious issues, hostile tone.
        Set confidence: high/medium/low based on clarity.
        """

        try:
            result = retry_agent_call(
                lambda: classifier.kickoff(query, response_format=IntentClassification)
            )
            classification = result.pydantic
            print(
                f"[CLASSIFY] Intent: {classification.intent} | Escalation: {classification.requires_escalation} | Confidence: {classification.confidence}"
            )
            self.state.classification = classification
        except Exception as e:
            print(f"[CLASSIFY] Critical error - routing to fallback: {str(e)}")
            # Fallback to general handler on critical failure
            return "fallback"

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

        try:
            result = retry_agent_call(
                lambda: menu_specialist.kickoff(query, response_format=MenuResponse)
            )
            print(f"[MENU] Found {len(result.pydantic.menu_items)} items")
            self.state.menu_response = result.pydantic
        except Exception as e:
            print(f"[MENU] Error processing menu inquiry: {str(e)}")
            # Create error response
            self.state.menu_response = MenuResponse(
                menu_items=["Unable to retrieve menu at this time. Please try again."]
            )

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

        context = self._get_context_summary()

        query = f"""
        CUSTOMER MESSAGE: '{self.state.customer_message}'
        
        CONTEXT: {context}

        CRITICAL: Read the customer message and context carefully.
        
        CONTEXT INTERPRETATION:
        - If context shows "last_order_id: X" and "recent_items: Y", the customer has a previous order
        - If customer says "my last order", "previous order", "that order", they're referring to last_order_id
        - If customer says "make it 2", "change to 2", "double it", they want to modify quantity
        - If customer says "add X", they want to add items to a new order
        
        For MODIFYING existing order (e.g., "make it 2 for my last order"):
        1. Check context for "last_order_id" and "recent_items"
        2. If found, use order_lookup(action='lookup_by_id', order_id=<last_order_id>) to get current order
        3. Explain that you found their previous order and what changes they want
        4. Set order_status to describe the modification request
        
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
        2. CRITICAL: order_id is NOT the same as customer_id!
           - order_id = the ID of a specific order (e.g., "order #18")
           - customer_id = the ID of a customer (from customer_lookup)
        3. If explicit order_id provided (e.g., "#18", "order 42"):
           call order_lookup(action='lookup_by_id', order_id=18)
        4. If phone OR name provided (NO explicit order_id):
           a. If phone given (e.g., "phone is 555-1005"): 
              call order_lookup(action='lookup_by_phone', phone="555-1005")
           b. If name given (e.g., "name is Mia Johnson", "for Mia Johnson"): 
              - First use customer_lookup(query="Mia Johnson") to get phone
              - Tool returns: "ID: 5 | Mia Johnson | Phone: 555-1005 | Email: mia.johnson@example.com"
              - Extract PHONE NUMBER (555-1005), NOT the customer ID!
              - Then call order_lookup(action='lookup_by_phone', phone="555-1005")
           - Tool output example:
             "Orders for Mia Johnson (phone: 555-1005):\n\nOrder #27 - in_progress\n  Date: 2025-12-01 18:30:00\n  Total: $45.00"
           - Use the most recent order entry to populate OrderResponse (order_id, items if provided, total_amount, order_status)
        
        After tool call: If JSON_SUMMARY present, parse it. Otherwise parse text output to fill OrderResponse fields.
        
        IMPORTANT: When filling OrderResponse:
        - items_ordered must be an ARRAY of objects, NOT a string
        - Each item should be: {{"menu": "Item Name", "price": "$X.XX", "quantity": N}}
        - Example: items_ordered = [{{"menu": "Caesar Salad", "price": "$10.00", "quantity": 2}}]
        - Do NOT wrap the array in quotes or escape characters
        
        Ensure reported items/status match customer request.
        """

        try:
            result = retry_agent_call(
                lambda: order_handler.kickoff(query, response_format=OrderResponse)
            )
            print(
                f"[ORDER] ID: {result.pydantic.order_id} | Items: {len(result.pydantic.items_ordered)} | Total: ${result.pydantic.total_amount} | Status: {result.pydantic.order_status}"
            )
            self.state.order_response = result.pydantic

            # Extract customer_id from message if not set
            if (
                not self.state.current_customer_id
                and "for" in self.state.customer_message.lower()
            ):
                # Try to get customer_id from customer_lookup
                import re

                name_match = re.search(
                    r"for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                    self.state.customer_message,
                )
                if name_match:
                    customer_name = name_match.group(1)
                    lookup_tool = CustomerLookupTool()
                    lookup_result = lookup_tool._run(query=customer_name)
                    id_match = re.search(r"ID:\s+(\d+)", lookup_result)
                    if id_match:
                        self.state.current_customer_id = int(id_match.group(1))
                        print(
                            f"[MEMORY] Set customer_id to {self.state.current_customer_id}"
                        )

            # Save order preferences to memory
            if result.pydantic.order_id:
                self._update_memory("last_order_id", str(result.pydantic.order_id))
            if result.pydantic.items_ordered:
                items = ", ".join([item.menu for item in result.pydantic.items_ordered])
                self._update_memory("recent_items", items)

        except Exception as e:
            print(f"[ORDER] Error processing order: {str(e)}")
            # Create error response
            self.state.order_response = OrderResponse(
                order_id=0,
                items_ordered=[],
                total_amount=0.0,
                order_status="error - unable to process request",
            )

        return self.state.order_response

    @listen("reservation_request")
    def handle_reservation(self):
        print(
            f"[RESERVATION] Processing reservation request: '{self.state.customer_message}'"
        )

        reservation_agent = Agent(
            role="Reservation Agent",
            goal="Manage reservations",
            backstory="Reservation specialist using reservation_lookup and customer_lookup tools.",
            tools=[ReservationLookupTool(), CustomerLookupTool()],
            verbose=True,
            llm=llm,
        )

        context = self._get_context_summary()

        query = f"""
        CUSTOMER MESSAGE: '{self.state.customer_message}'
        
        CONTEXT: {context}

        CRITICAL: Read the customer message carefully. Use ONLY the information provided in the message.
        Use context to fill in missing details (e.g., if customer says "change to 8pm" and context shows previous reservation).
        
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
        2. CRITICAL: reservation_id is NOT the same as customer_id!
           - reservation_id = the ID of a specific reservation (e.g., "reservation #15")
           - customer_id = the ID of a customer (from customer_lookup)
        3. If explicit reservation_id provided (e.g., "reservation #15", "booking ID 20"):
           call reservation_lookup(action='lookup_by_id', reservation_id=15)
        4. If phone OR name provided (NO explicit reservation_id):
           a. If phone given (e.g., "phone is 555-1005"): 
              call reservation_lookup(action='lookup_by_phone', phone="555-1005")
           b. If name given (e.g., "name is Liam Patel", "for Liam Patel"):
              - First use customer_lookup(query="Liam Patel") to get phone
              - Tool returns: "ID: 2 | Liam Patel | Phone: 555-1004 | Email: liam.patel@example.com"
              - Extract PHONE NUMBER (555-1004), NOT the customer ID!
              - Then call reservation_lookup(action='lookup_by_phone', phone="555-1004")
           - Tool output looks like:
             "Reservations for Mia Johnson (phone: 555-1005):\n\nReservation #15 - confirmed\n  Date & Time: 2025-12-04 19:00:00\n  Party Size: 6 people"
           - Parse the first reservation entry (closest upcoming) to fill ReservationResponse:
             reservation_id=15, party_size=6, reservation_datetime="2025-12-04 19:00:00", status="confirmed"
        
        Fill ReservationResponse fields with the parsed data. If no reservations returned, set status="awaiting_details" and explain that no reservation exists.
        """

        try:
            result = retry_agent_call(
                lambda: reservation_agent.kickoff(
                    query, response_format=ReservationResponse
                )
            )
            print(
                f"[RESERVATION] ID: {result.pydantic.reservation_id} | Party: {result.pydantic.party_size} | Time: {result.pydantic.reservation_datetime} | Status: {result.pydantic.status}"
            )
            self.state.reservation_response = result.pydantic

            # Extract customer_id from message if not set
            if (
                not self.state.current_customer_id
                and "for" in self.state.customer_message.lower()
            ):
                # Try to get customer_id from customer_lookup
                import re

                name_match = re.search(
                    r"for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                    self.state.customer_message,
                )
                if name_match:
                    customer_name = name_match.group(1)
                    lookup_tool = CustomerLookupTool()
                    lookup_result = lookup_tool._run(query=customer_name)
                    id_match = re.search(r"ID:\s+(\d+)", lookup_result)
                    if id_match:
                        self.state.current_customer_id = int(id_match.group(1))
                        print(
                            f"[MEMORY] Set customer_id to {self.state.current_customer_id}"
                        )

            # Save reservation preferences to memory
            if result.pydantic.reservation_id:
                self._update_memory(
                    "last_reservation_id", str(result.pydantic.reservation_id)
                )
            if result.pydantic.party_size:
                self._update_memory("usual_party_size", str(result.pydantic.party_size))
            if result.pydantic.reservation_datetime:
                self._update_memory(
                    "last_reservation_time", result.pydantic.reservation_datetime
                )

        except Exception as e:
            print(f"[RESERVATION] Error processing reservation: {str(e)}")
            # Create error response
            self.state.reservation_response = ReservationResponse(
                reservation_id=0,
                party_size=0,
                reservation_datetime="N/A",
                status="error - unable to process request",
                special_requests="",
            )

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

        try:
            result = retry_agent_call(
                lambda: composer.kickoff(query, response_format=FinalResponse)
            )
            print("[DELIVER] Final response composed")
            self.state.final_response = result.pydantic
        except Exception as e:
            print(f"[DELIVER] Error composing response: {str(e)}")
            # Create fallback response
            self.state.final_response = FinalResponse(
                customer_message_summary="Unable to process request",
                final_response="We apologize, but we're experiencing technical difficulties. Please try again in a moment or contact us directly for assistance.",
            )

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
