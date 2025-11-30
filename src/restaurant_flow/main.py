#!/usr/bin/env python
from typing import Any, Dict

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from restaurant_flow.crews.restaurant_crew.restaurant_crew import RestaurantCrew


class RestaurantState(BaseModel):
    customer_message: str = "What vegan options do you have?"
    response: str = ""


class RestaurantFlow(Flow[RestaurantState]):
    @start()
    def receive_customer_message(self, crewai_trigger_payload: Dict[str, Any] | None = None):
        """Capture the guest's question when the flow starts."""
        print("Receiving customer message")

        if crewai_trigger_payload and crewai_trigger_payload.get("customer_message"):
            message = str(crewai_trigger_payload["customer_message"])
            print("Using trigger payload for customer message")
        else:
            message = self.state.customer_message
            print("Using default customer message")

        self.state.customer_message = message
        print(f"Customer message: {self.state.customer_message}")

    @listen(receive_customer_message)
    def answer_menu_question(self):
        """Ask the menu specialist crew to answer the guest's question."""
        print("Routing question to menu specialist crew")

        crew = RestaurantCrew().crew()
        result = crew.kickoff(inputs={"customer_message": self.state.customer_message})

        self.state.response = result.raw
        print("Menu response ready")

    @listen(answer_menu_question)
    def summarize_response(self):
        """Output the final response for downstream use."""
        print("Final menu response:\n")
        print(self.state.response)


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
    kickoff()
