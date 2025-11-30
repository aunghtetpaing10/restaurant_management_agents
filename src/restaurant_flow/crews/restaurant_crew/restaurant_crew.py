from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from restaurant_flow.tools import MenuSearchTool


@CrewBase
class RestaurantCrew:
    """RestaurantCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def menu_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["menu_specialist"],  # type: ignore[index]
            tools=[MenuSearchTool()],
            verbose=True,
        )

    @task
    def menu_qna_task(self) -> Task:
        return Task(
            config=self.tasks_config["menu_qna_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the RestaurantCrew crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
