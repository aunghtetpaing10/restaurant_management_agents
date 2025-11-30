from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from restaurant_flow.tools import (
    MenuSearchTool,
    OrderLookupTool,
    ReservationLookupTool,
)


@CrewBase
class RestaurantCrew:
    """RestaurantCrew crew for handling customer requests"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def intent_classifier(self) -> Agent:
        return Agent(
            config=self.agents_config["intent_classifier"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def menu_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["menu_specialist"],  # type: ignore[index]
            tools=[MenuSearchTool()],
            verbose=True,
        )

    @agent
    def order_handler(self) -> Agent:
        return Agent(
            config=self.agents_config["order_handler"],  # type: ignore[index]
            tools=[MenuSearchTool(), OrderLookupTool()],
            verbose=True,
        )

    @agent
    def reservation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["reservation_agent"],  # type: ignore[index]
            tools=[ReservationLookupTool()],
            verbose=True,
        )

    @agent
    def response_composer(self) -> Agent:
        return Agent(
            config=self.agents_config["response_composer"],  # type: ignore[index]
            verbose=True,
        )

    @task
    def classify_intent_task(self) -> Task:
        return Task(
            config=self.tasks_config["classify_intent_task"],  # type: ignore[index]
        )

    @task
    def menu_inquiry_task(self) -> Task:
        return Task(
            config=self.tasks_config["menu_inquiry_task"],  # type: ignore[index]
        )

    @task
    def process_order_task(self) -> Task:
        return Task(
            config=self.tasks_config["process_order_task"],  # type: ignore[index]
        )

    @task
    def handle_reservation_task(self) -> Task:
        return Task(
            config=self.tasks_config["handle_reservation_task"],  # type: ignore[index]
        )

    @task
    def compose_response_task(self) -> Task:
        return Task(
            config=self.tasks_config["compose_response_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the RestaurantCrew with all agents in sequential process"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
