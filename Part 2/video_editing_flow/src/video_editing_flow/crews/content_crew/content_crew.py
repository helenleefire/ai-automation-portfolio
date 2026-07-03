from crewai import Agent, Crew, Process, Task, LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from video_editing_flow.tools.custom_tool import ContentPlan

llm = LLM(model="anthropic/claude-sonnet-4-6")


@CrewBase
class ContentCrew:
    """Content Crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def content_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["content_planner"],  # type: ignore[index]
            llm=llm,
        )

    def content_planning_task(self, strategy: str = "visual") -> Task:
        task_key = "content_planning_task_audio_driven" if strategy == "audio" else "content_planning_task_visual_driven"
        return Task(
            config=self.tasks_config[task_key],  # type: ignore[index]
            output_pydantic=ContentPlan,
        )

    def crew(self, strategy: str = "visual") -> Crew:
        return Crew(
            agents=[self.content_planner()],  # type: ignore[misc]
            tasks=[self.content_planning_task(strategy)],  # type: ignore[misc]
            process=Process.sequential,
            verbose=True,
        )
