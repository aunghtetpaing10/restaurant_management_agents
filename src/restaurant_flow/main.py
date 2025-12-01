from typing import List
from crewai import LLM, Agent
from crewai.flow.flow import Flow, listen, or_, start, router
from pydantic import BaseModel, Field

from restaurant_flow.tools.custom_tool import (
    MenuSearchTool,
    OrderLookupTool,
    ReservationLookupTool,
)

llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434"
)


class IntentClassification(BaseModel):
    """Intent classifier structured output."""

    intent: str = Field(
        description="Detected intent category (menu_inquiry, order_request, reservation_request, general_question, complaint, unclear, other)"
    )
    requires_escalation: bool = Field(
        default=False,
        description="True if the message contains a complaint, urgent issue, or requires human intervention"
    )
    confidence: str = Field(
        default="high",
        description="Confidence level in the classification: high, medium, or low"
    )


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

    order_id: int | None = Field(
        default=None,
        description="ID of the order (if known or after creation)",
    )
    items_ordered: List[str] = Field(
        description="List of ordered items (e.g., ['Caesar Salad', 'Margherita Pizza'])"
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
    customer_message: str = "How do I order a meal?"
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
            goal="Accurately determine the customer's underlying intent",
            backstory=(
                "You are a restaurant service triage specialist. You quickly understand what the customer really wants from their message and route them to the best specialist."
            ),
            verbose=True,
            llm=llm,
        )

        query = f"""
        Customer message: "{self.state.customer_message}"

        Your task is to analyze the customer's message and determine:
        1. The primary intent category
        2. Whether the message requires escalation (complaints, urgent issues, dissatisfaction)
        3. Your confidence level in the classification

        IMPORTANT: Read the message carefully and distinguish between ASKING ABOUT something vs. DOING something.

        Possible intents:
        - menu_inquiry: The guest wants menu details, recommendations, dietary info, prices. Examples: "What's on the menu?", "Do you have vegan options?", "How much is the steak?"
        
        - order_request: The guest is ACTIVELY placing an order RIGHT NOW with specific items, OR checking/modifying an existing order with an order ID/number. 
          Examples: "I want to order 2 chicken wings", "I'll have the Caesar salad", "Where is my order #123?", "Cancel my order"
          NOT THIS: "How do I order?", "What's the ordering process?", "Can I order online?"
        
        - reservation_request: The guest is ACTIVELY making a reservation RIGHT NOW with specific details (date/time/party size), OR checking/modifying an existing reservation.
          Examples: "Book a table for 4 at 7pm tomorrow", "I'd like to reserve for tonight", "Cancel my reservation for Smith"
          NOT THIS: "How do I make a reservation?", "Do you take reservations?", "What's your booking policy?"
        
        - general_question: The guest is asking HOW to do something, asking about policies/procedures/hours/location, or seeking information WITHOUT taking action.
          Examples: "How do I order a meal?", "How can I make a reservation?", "What are your hours?", "Do you deliver?", "Where are you located?", "How does payment work?"
          This is for INFORMATIONAL questions, not action requests.
        
        - complaint: The guest is expressing dissatisfaction, reporting a problem, or making a complaint.
        
        - unclear: The message is ambiguous, incomplete, or you cannot confidently determine the intent.
        
        - other: Anything that does not fit the above categories.

        Escalation criteria (set requires_escalation=true if ANY of these apply):
        - Customer is complaining about service, food quality, wait times, or any aspect of their experience
        - Customer is expressing anger, frustration, or strong dissatisfaction
        - Customer is requesting to speak with a manager or supervisor
        - Customer is threatening negative reviews, legal action, or similar escalations
        - Customer is reporting a serious issue (food safety, allergies not honored, billing errors)
        - Message tone is hostile, aggressive, or demanding immediate attention

        Confidence guidelines:
        - high: The intent is clear and unambiguous
        - medium: The intent is likely but could have alternative interpretations
        - low: The message is vague, unclear, or could fit multiple categories

        DECISION TREE:
        1. Does the message contain "how", "how do I", "how can I", "what's the process", "can I", or similar question words about procedures?
           → Likely general_question
        
        2. Does the message specify items to order (e.g., "chicken wings", "salad", "pizza") OR an order number?
           → Likely order_request
        
        3. Does the message specify reservation details (date, time, party size) OR a reservation name/ID?
           → Likely reservation_request
        
        4. Does the message ask about menu items, prices, ingredients, or food options?
           → Likely menu_inquiry

        CRITICAL: "How do I order a meal?" is asking ABOUT the ordering process = general_question, NOT order_request.

        Provide your classification with the intent, escalation flag, and confidence level.
        """

        result = classifier.kickoff(query, response_format=IntentClassification)
        classification = result.pydantic

        print("\n" + "=" * 60)
        print("[DEBUG] Intent Classification Result:")
        print(f"  - Intent: {classification.intent}")
        print(f"  - Requires Escalation: {classification.requires_escalation}")
        print(f"  - Confidence: {classification.confidence}")
        print("=" * 60)

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
            goal="Provide detailed, accurate menu information",
            backstory=(
                "You are a knowledgeable menu expert who knows every dish, ingredient, "
                "and dietary option. You use the menu_search tool to query the database."
            ),
            tools=[MenuSearchTool()],
            verbose=True,
            llm=llm,
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

    @listen("order_request")
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
            llm=llm,
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

        ORDER INTAKE MODE (no explicit order ID detected)

        Your task:
        1. Determine if the guest is trying to PLACE a NEW ORDER.
           - Look for menu item names or ordering language ("I'd like", "Can I order", etc.).
           - Reference the menu cheat sheet below when mapping items to menu_item_id values.
        2. Extract the requested items with quantities. If quantity not given, assume 1 and note that clarification is needed.
        3. Check for any customer identifiers (name, phone, email, loyalty ID). If none provided, note that they are needed before order creation.
        4. Only call the order_lookup tool with action 'create' if ALL required data is present:
             - action: 'create'
             - customer_id: <ID from message or known mapping>
             - items: JSON string like [{{"menu_item_id": 5, "quantity": 2}}, ...]
           If required data is missing, DO NOT call the tool. Instead, set order_status to "awaiting_customer_details" and list what is missing.
        5. Populate the OrderResponse as follows:
             - order_id: Use ID returned by the tool if created; otherwise leave as null.
             - items_ordered: List the items the guest wants (even if the order isn’t created yet).
             - total_amount: Use value from the tool if available; otherwise 0.0 until confirmed.
             - order_status: One of "created", "awaiting_customer_details", "unrecognized_request", or another clear status.
        6. If the message is NOT an order request, set order_status to "unrecognized_request" and explain why in the reasoning.
        7. Be explicit in your reasoning so downstream agents know the current state of the order.

        MENU CHEAT SHEET (ID -> Item Name):
        - 2 -> "Crispy Calamari"
        - 42 -> "Cold Brew Coffee"

        TOOL USAGE EXAMPLE (only when all required data is present):
        order_lookup(action='create', customer_id=1, items='[{{"menu_item_id": 2, "quantity": 1}}, {{"menu_item_id": 42, "quantity": 1}}]')
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

    @listen("reservation_request")
    def handle_reservation(self):
        print("[RESERVATION] Processing reservation request...")

        reservation_agent = Agent(
            role="Reservation Agent",
            goal="Manage table reservations accurately",
            backstory=(
                "You coordinate reservations, ensure availability, and communicate clearly with guests."
            ),
            tools=[ReservationLookupTool()],
            verbose=True,
            llm=llm,
        )

        query = f"""
        Customer request: '{self.state.customer_message}'

        You are the restaurant's reservation specialist. Determine how best to help this guest:
        1. Identify whether they are referencing an existing reservation (modifying, confirming, cancelling) or requesting a brand-new booking.
        2. Choose the appropriate reservation_lookup action:
           - reservation_lookup(action='lookup_by_id', reservation_id=123) when a reservation number is provided.
           - reservation_lookup(action='lookup_by_phone', phone='5551234567') when only a phone/contact is provided.
           - reservation_lookup(action='create', customer_id=4, party_size=2, reservation_date='2025-12-24', reservation_time='19:30') when you have enough details to book a new table AND a valid customer identifier (ID or a mapped phone number).
        3. If critical details (party size, date, time, customer identifier) are missing, DO NOT call the tool. Instead, set status to "awaiting_details" and clearly list what is needed in special_requests.
        3a. If customer_id is missing but the guest provided a phone number, first run reservation_lookup(action='lookup_by_phone', ...) to retrieve the customer record and use its ID when creating the reservation.
        4. Populate ReservationResponse carefully:
           - reservation_id: from the tool response when available (leave null otherwise).
           - party_size: numeric value inferred from the message or tool.
           - reservation_datetime: use "YYYY-MM-DD HH:MM" when you know both; otherwise provide the best-available description (e.g., "tomorrow 7pm").
           - status: choose a clear lifecycle label such as confirmed, awaiting_details, not_found, cancelled, etc.
           - special_requests: include guest preferences, accessibility notes, or outstanding questions for follow-up.
        5. Keep your reasoning concise and professional so the concierge team can act immediately on your output.
        """

        result = reservation_agent.kickoff(query, response_format=ReservationResponse)

        print("\n" + "=" * 60)
        print("[DEBUG] Reservation Agent Result:")
        print(f"  - Reservation ID: {result.pydantic.reservation_id}")
        print(f"  - Party size: {result.pydantic.party_size}")
        print(f"  - Date/time: {result.pydantic.reservation_datetime}")
        print(f"  - Status: {result.pydantic.status}")
        print(f"  - Special requests: {result.pydantic.special_requests}")
        print("=" * 60)

        self.state.reservation_response = result.pydantic
        return self.state.reservation_response

    @listen("escalation")
    def handle_escalation(self):
        print("[ESCALATION] Processing escalation request...")

        escalation_agent = Agent(
            role="Customer Service Manager",
            goal="Handle escalations and complaints with empathy and urgency",
            backstory=(
                "You are a senior customer service manager who specializes in de-escalating situations, "
                "addressing complaints professionally, and ensuring customers feel heard and valued. "
                "You have the authority to offer solutions and take immediate action."
            ),
            verbose=True,
            llm=llm,
        )

        query = f"""
        Customer message: '{self.state.customer_message}'
        Classification: {self.state.classification.intent}
        Escalation reason: This message was flagged as requiring immediate attention.

        Your task:
        1. Acknowledge the customer's concern with empathy and professionalism.
        2. Apologize sincerely if appropriate.
        3. Explain what immediate steps you will take to address their issue.
        4. Offer a specific solution, compensation, or next steps.
        5. Provide direct contact information for follow-up (manager line, email).
        6. Ensure the tone is warm, understanding, and solution-focused.

        Create a response that de-escalates the situation and shows the customer we take their concern seriously.
        """

        result = escalation_agent.kickoff(query, response_format=FinalResponse)

        print("\n" + "=" * 60)
        print("[ESCALATION] Manager Response:")
        print(result.pydantic.final_response)
        print("=" * 60)

        self.state.final_response = result.pydantic
        return self.state.final_response

    @listen("fallback")
    def handle_fallback(self):
        print("[FALLBACK] Processing unclear or general request...")

        fallback_agent = Agent(
            role="General Support Agent",
            goal="Assist with general inquiries and clarify unclear requests",
            backstory=(
                "You are a friendly and helpful support agent who assists customers when their request "
                "doesn't fit a specific category or needs clarification. You ask clarifying questions "
                "and guide customers to the right service."
            ),
            verbose=True,
            llm=llm,
        )

        query = f"""
        Customer message: '{self.state.customer_message}'
        Classification: {self.state.classification.intent}
        Confidence: {self.state.classification.confidence}

        Your task:
        1. If the intent is unclear, politely ask clarifying questions to understand what the customer needs.
        2. If it's a general question, provide helpful information or guide them to the right department.
        3. Offer specific options: "Would you like to see our menu, place an order, or make a reservation?"
        4. Keep the tone friendly, patient, and helpful.
        5. If you can partially address their request, do so while asking for clarification on unclear parts.

        Create a response that helps the customer get what they need or guides them to provide more information.
        """

        result = fallback_agent.kickoff(query, response_format=FinalResponse)

        print("\n" + "=" * 60)
        print("[FALLBACK] Support Response:")
        print(result.pydantic.final_response)
        print("=" * 60)

        self.state.final_response = result.pydantic
        return self.state.final_response

    @listen(or_(handle_menu, handle_order, handle_reservation))
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
            specialist_data = f"""
            Order Handler Response:
            - Order ID: {order.order_id}
            - Items ordered: {", ".join(order.items_ordered)}
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
