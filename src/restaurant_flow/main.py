import time

from crewai.flow.flow import Flow, listen, or_, start, router

from restaurant_flow.models import (
    IntentClassification,
    MenuResponse,
    OrderResponse,
    ReservationResponse,
    FinalResponse,
    RestaurantState,
    MemoryKeys,
)
from restaurant_flow.prompts import (
    get_intent_classification_prompt,
    get_menu_inquiry_prompt,
    get_order_handler_prompt,
    get_reservation_handler_prompt,
    get_escalation_prompt,
    get_fallback_prompt,
    get_response_composer_prompt,
)
from restaurant_flow.agents import (
    create_intent_classifier,
    create_menu_specialist,
    create_order_handler,
    create_reservation_agent,
    create_escalation_agent,
    create_fallback_agent,
    create_response_composer,
)
from restaurant_flow.tools.custom_tool import CustomerLookupTool
from restaurant_flow.tools.preference_tools import CustomerPreferenceTool


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


class RestaurantFlow(Flow[RestaurantState]):
    # Dietary keywords to detect in messages
    DIETARY_KEYWORDS = {
        "vegetarian": "vegetarian",
        "vegan": "vegan",
        "gluten-free": "gluten-free",
        "gluten free": "gluten-free",
        "dairy-free": "dairy-free",
        "dairy free": "dairy-free",
        "halal": "halal",
        "kosher": "kosher",
        "pescatarian": "pescatarian",
        "keto": "keto",
        "low-carb": "low-carb",
    }
    
    ALLERGY_KEYWORDS = {
        "nut allergy": "nuts",
        "peanut allergy": "peanuts",
        "allergic to nuts": "nuts",
        "allergic to peanuts": "peanuts",
        "allergic to shellfish": "shellfish",
        "shellfish allergy": "shellfish",
        "allergic to dairy": "dairy",
        "lactose intolerant": "dairy",
        "allergic to gluten": "gluten",
        "celiac": "gluten",
        "allergic to eggs": "eggs",
        "egg allergy": "eggs",
        "allergic to soy": "soy",
        "soy allergy": "soy",
    }

    def _extract_customer_id(self) -> int | None:
        """Extract customer ID from message if customer name is mentioned.

        Searches for patterns like 'for Noah Chen' and looks up the customer in the database.
        Returns the customer ID if found, None otherwise.
        """
        import re

        if "for" not in self.state.customer_message.lower():
            return None

        name_match = re.search(
            r"for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            self.state.customer_message,
        )
        if not name_match:
            return None

        customer_name = name_match.group(1)
        lookup_tool = CustomerLookupTool()
        lookup_result = lookup_tool._run(query=customer_name)

        id_match = re.search(r"ID:\s+(\d+)", lookup_result)
        if id_match:
            customer_id = int(id_match.group(1))
            print(f"[MEMORY] Identified customer_id: {customer_id} ({customer_name})")
            return customer_id

        return None

    def _extract_dietary_info(self) -> tuple[list[str], list[str]]:
        """Extract dietary restrictions and allergies from customer message.
        
        Returns tuple of (dietary_restrictions, allergies).
        """
        message_lower = self.state.customer_message.lower()
        
        # Detect dietary restrictions
        dietary = []
        for keyword, normalized in self.DIETARY_KEYWORDS.items():
            if keyword in message_lower and normalized not in dietary:
                dietary.append(normalized)
        
        # Detect allergies
        allergies = []
        for keyword, normalized in self.ALLERGY_KEYWORDS.items():
            if keyword in message_lower and normalized not in allergies:
                allergies.append(normalized)
        
        return dietary, allergies

    def _save_dietary_info(self):
        """Detect and save dietary info from current message."""
        if not self.state.current_customer_id:
            return
            
        dietary, allergies = self._extract_dietary_info()
        
        if dietary:
            self._update_memory(MemoryKeys.DIETARY_RESTRICTIONS, ", ".join(dietary))
            print(f"[MEMORY] Detected dietary restrictions: {dietary}")
            
        if allergies:
            self._update_memory(MemoryKeys.ALLERGIES, ", ".join(allergies))
            print(f"[MEMORY] Detected allergies: {allergies}")

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
        """Get customer preferences from database in agent-friendly format."""
        if not self.state.current_customer_id:
            return "New customer - no previous history."

        # Get preferences from database
        pref_tool = CustomerPreferenceTool()
        prefs_result = pref_tool._run(
            action="get_all", customer_id=self.state.current_customer_id
        )

        if prefs_result.startswith("No preferences"):
            return "New customer - no previous history."

        # Parse and format as structured context
        context_lines = ["Customer History:"]
        for line in prefs_result.split("\n"):
            if ":" in line and not line.startswith("Preferences"):
                key, value = line.strip().split(":", 1)
                key = key.strip()
                value = value.strip()
                
                # Format based on key type for better agent understanding
                if key == MemoryKeys.LAST_ORDER_ID:
                    context_lines.append(f"- Last order ID: #{value}")
                elif key == MemoryKeys.RECENT_ITEMS:
                    context_lines.append(f"- Recently ordered: {value}")
                elif key == MemoryKeys.LAST_RESERVATION_ID:
                    context_lines.append(f"- Last reservation ID: #{value}")
                elif key == MemoryKeys.USUAL_PARTY_SIZE:
                    context_lines.append(f"- Usual party size: {value} people")
                elif key == MemoryKeys.RECENT_MENU_SEARCHES:
                    context_lines.append(f"- Recently browsed: {value}")
                elif key == MemoryKeys.DIETARY_RESTRICTIONS:
                    context_lines.append(f"- Dietary restrictions: {value}")
                elif key == MemoryKeys.ALLERGIES:
                    context_lines.append(f"- Allergies: {value}")
                else:
                    context_lines.append(f"- {key}: {value}")

        return "\n".join(context_lines) if len(context_lines) > 1 else "New customer - no previous history."

    def _format_specialist_data(self) -> str:
        """Format specialist response data for the response composer."""
        if self.state.menu_response:
            menu = self.state.menu_response
            return f"""
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

            return f"""
            Order Handler Response:
            - Order ID: {order.order_id}
            - Items: {order_items_lines}
            - Total amount: ${order.total_amount}
            - Order status: {order.order_status}
            """
        elif self.state.reservation_response:
            reservation = self.state.reservation_response
            return f"""
            Reservation Agent Response:
            - Reservation ID: {reservation.reservation_id}
            - Party size: {reservation.party_size}
            - Date: {reservation.reservation_datetime}
            - Status: {reservation.status}
            - Special requests: {reservation.special_requests}
            """
        return ""

    @start()
    def receive_message(self):
        """Entry point for the flow. Extracts customer ID and dietary info if mentioned."""
        print("Starting the structured flow")

        # Extract customer_id early if customer name is mentioned
        if not self.state.current_customer_id:
            self.state.current_customer_id = self._extract_customer_id()

        # Detect and save dietary preferences/allergies from message
        self._save_dietary_info()

    @router(receive_message)
    def classify_intent(self):
        """Classify customer intent and route to appropriate handler."""
        print("[CLASSIFY] Classifying intent with agent...")

        classifier = create_intent_classifier()
        context = self._get_context_summary()
        query = get_intent_classification_prompt(self.state.customer_message, context)

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
        """Handle menu inquiry requests."""
        print("[MENU] Processing menu inquiry...")

        menu_specialist = create_menu_specialist()
        context = self._get_context_summary()
        query = get_menu_inquiry_prompt(self.state.customer_message, context)

        try:
            result = retry_agent_call(
                lambda: menu_specialist.kickoff(query, response_format=MenuResponse)
            )
            print(f"[MENU] Found {len(result.pydantic.menu_items)} items")
            self.state.menu_response = result.pydantic

            # Save menu search to memory (track what customer is interested in)
            if self.state.current_customer_id and result.pydantic.menu_items:
                recent_searches = ", ".join(result.pydantic.menu_items[:3])
                self._update_memory(MemoryKeys.RECENT_MENU_SEARCHES, recent_searches)

        except Exception as e:
            print(f"[MENU] Error processing menu inquiry: {str(e)}")
            # Create error response
            self.state.menu_response = MenuResponse(
                menu_items=["Unable to retrieve menu at this time. Please try again."]
            )

        return self.state.menu_response

    @listen("order_request")
    def handle_order(self):
        """Handle order creation and lookup requests."""
        print(f"[ORDER] Processing order request: '{self.state.customer_message}'")

        order_handler = create_order_handler()
        context = self._get_context_summary()
        query = get_order_handler_prompt(self.state.customer_message, context)

        try:
            result = retry_agent_call(
                lambda: order_handler.kickoff(query, response_format=OrderResponse)
            )
            print(
                f"[ORDER] ID: {result.pydantic.order_id} | Items: {len(result.pydantic.items_ordered)} | Total: ${result.pydantic.total_amount} | Status: {result.pydantic.order_status}"
            )
            self.state.order_response = result.pydantic

            # Extract customer_id from message if not set
            if not self.state.current_customer_id:
                self.state.current_customer_id = self._extract_customer_id()

            # Save order preferences to memory
            if result.pydantic.order_id:
                self._update_memory(MemoryKeys.LAST_ORDER_ID, str(result.pydantic.order_id))
            if result.pydantic.items_ordered:
                items = ", ".join([item.menu for item in result.pydantic.items_ordered])
                self._update_memory(MemoryKeys.RECENT_ITEMS, items)

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
        """Handle reservation creation and lookup requests."""
        print(
            f"[RESERVATION] Processing reservation request: '{self.state.customer_message}'"
        )

        reservation_agent = create_reservation_agent()
        context = self._get_context_summary()
        query = get_reservation_handler_prompt(self.state.customer_message, context)

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
            if not self.state.current_customer_id:
                self.state.current_customer_id = self._extract_customer_id()

            # Save reservation preferences to memory
            if result.pydantic.reservation_id:
                self._update_memory(
                    MemoryKeys.LAST_RESERVATION_ID, str(result.pydantic.reservation_id)
                )
            if result.pydantic.party_size:
                self._update_memory(MemoryKeys.USUAL_PARTY_SIZE, str(result.pydantic.party_size))
            if result.pydantic.reservation_datetime:
                self._update_memory(
                    MemoryKeys.LAST_RESERVATION_TIME, result.pydantic.reservation_datetime
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
        """Handle escalation requests with empathy."""
        print("[ESCALATION] Processing escalation request...")

        escalation_agent = create_escalation_agent()
        query = get_escalation_prompt(
            self.state.customer_message, self.state.classification.intent
        )

        result = escalation_agent.kickoff(query, response_format=FinalResponse)

        print("[ESCALATION] Response generated")

        self.state.final_response = result.pydantic
        return self.state.final_response

    @listen("fallback")
    def handle_fallback(self):
        """Handle unclear or general requests."""
        print("[FALLBACK] Processing unclear or general request...")

        fallback_agent = create_fallback_agent()
        query = get_fallback_prompt(
            self.state.customer_message,
            self.state.classification.intent,
            self.state.classification.confidence,
        )

        result = fallback_agent.kickoff(query, response_format=FinalResponse)

        print("[FALLBACK] Response generated")

        self.state.final_response = result.pydantic
        return self.state.final_response

    @listen(or_(handle_menu, handle_order, handle_reservation))
    def deliver_response(self):
        """Compose the final polished response to the customer."""
        composer = create_response_composer()

        # Format the specialist response data for the composer
        specialist_data = self._format_specialist_data()
        query = get_response_composer_prompt(
            self.state.customer_message, specialist_data
        )

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
    """Run the restaurant flow with proper MCP cleanup (single-shot mode)."""
    from restaurant_flow.mcp_init import close_mcp_tools

    try:
        restaurant_flow = RestaurantFlow()
        restaurant_flow.kickoff()
    finally:
        close_mcp_tools()


def chat():
    """Run interactive multi-turn chat mode."""
    from restaurant_flow.mcp_init import close_mcp_tools
    from restaurant_flow.conversation import ConversationManager

    manager = ConversationManager()
    session_id = "interactive"

    print("=" * 60)
    print("Restaurant Assistant (Multi-turn Chat)")
    print("Type 'quit' to exit, 'reset' to start over")
    print("=" * 60)

    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "reset":
                manager.reset_session(session_id)
                print("Session reset. Starting fresh!")
                continue
            
            if not user_input:
                continue

            response = manager.chat(user_input, session_id)
            print(f"\nAssistant: {response}")

    finally:
        close_mcp_tools()


def plot():
    """Generate a visual plot of the flow."""
    restaurant_flow = RestaurantFlow()
    restaurant_flow.plot()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat()
    else:
        kickoff()
