#!/usr/bin/env python
from typing import Any, Dict

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from restaurant_flow.crews.restaurant_crew.restaurant_crew import RestaurantCrew


class RestaurantState(BaseModel):
    """State for the restaurant flow containing customer message and response."""
    customer_message: str = "What's the status of order #18?"
    response: str = ""


class RestaurantFlow(Flow[RestaurantState]):
    """
    Restaurant management flow that handles customer requests through a multi-agent crew.
    
    The flow processes customer messages through 5 specialized agents:
    1. Intent Classifier - Determines what the customer wants
    2. Menu Specialist - Handles menu inquiries
    3. Order Handler - Processes orders and checks order status
    4. Reservation Agent - Manages reservations and checks reservation status
    5. Response Composer - Creates the final customer response
    """

    @start()
    def receive_customer_message(
        self, crewai_trigger_payload: Dict[str, Any] | None = None
    ):
        """Capture the customer's message when the flow starts."""
        print("🌊 Restaurant Flow Started")
        print("=" * 60)

        if crewai_trigger_payload and crewai_trigger_payload.get("customer_message"):
            message = str(crewai_trigger_payload["customer_message"])
            print("📥 Using trigger payload for customer message")
        else:
            message = self.state.customer_message
            print("📝 Using default customer message")

        self.state.customer_message = message
        print(f"💬 Customer message: {self.state.customer_message}")
        print("=" * 60)

    @listen(receive_customer_message)
    def process_customer_request(self):
        """
        Route the customer request through the multi-agent crew.
        
        The crew will:
        - Classify the customer's intent
        - Query menu database if needed
        - Process orders or check order status if needed
        - Handle reservations or check reservation status if needed
        - Compose a professional response
        """
        print("\n🤖 Routing request to multi-agent crew...")
        print("Agents: Intent Classifier → Menu Specialist → Order Handler → Reservation Agent → Response Composer")

        crew = RestaurantCrew().crew()
        result = crew.kickoff(inputs={"customer_message": self.state.customer_message})

        self.state.response = result.raw
        print("\n✅ Crew processing complete")

    @listen(process_customer_request)
    def deliver_response(self):
        """Output the final customer response."""
        print("\n" + "=" * 60)
        print("📤 FINAL CUSTOMER RESPONSE:")
        print("=" * 60)
        print(self.state.response)
        print("=" * 60)


def kickoff():
    restaurant_flow = RestaurantFlow()
    restaurant_flow.kickoff()


def plot():
    restaurant_flow = RestaurantFlow()
    restaurant_flow.plot()


def run_with_trigger():
    """
    Run the flow with trigger payload.
    """
    import json
    import sys

    # Get trigger payload from command line argument
    if len(sys.argv) < 2:
        raise Exception(
            "No trigger payload provided. Please provide JSON payload as argument."
        )

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    # Create flow and kickoff with trigger payload
    # The @start() methods will automatically receive crewai_trigger_payload parameter
    restaurant_flow = RestaurantFlow()

    try:
        result = restaurant_flow.kickoff({"crewai_trigger_payload": trigger_payload})
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the flow with trigger: {e}")


if __name__ == "__main__":
    """
    Run the restaurant flow with the default message.
    
    Example customer messages the system can handle:
    - "What vegan options do you have?" (Menu inquiry)
    - "I'd like to order 2 pizzas" (Order creation)
    - "What's the status of order #42?" (Order lookup)
    - "Book a table for 4 on Friday at 7pm" (Reservation creation)
    - "Check my reservations for phone 555-1234" (Reservation lookup)
    - "What's on the menu? Also book a table for 2" (Multiple intents)
    """
    kickoff()
