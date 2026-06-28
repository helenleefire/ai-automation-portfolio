from crewai import Agent, Crew, Process, Task, LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from video_editing_flow.tools.custom_tool import SceneScoringTool, ContentPlan, ScoredScenes

llm = LLM(model="anthropic/claude-sonnet-4-6")


@CrewBase
class ContentCrew:
    """Content Crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def scene_scorer(self) -> Agent:
        return Agent(
            config=self.agents_config["scene_scorer"],  # type: ignore[index]
            llm=llm,
        )

    @task
    def scene_scoring_task(self) -> Task:
        return Task(
            config=self.tasks_config["scene_scoring_task"],  # type: ignore[index]
            tools=[SceneScoringTool()],
            output_pydantic=ScoredScenes,
        )

    @agent
    def content_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["content_planner"],  # type: ignore[index]
            llm=llm,
        )

    @task
    def content_planning_task(self) -> Task:
        return Task(
            config=self.tasks_config["content_planning_task"],  # type: ignore[index]
            context=[self.scene_scoring_task()],  # type: ignore[misc]
            output_pydantic=ContentPlan,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Content Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
