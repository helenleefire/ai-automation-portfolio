from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from tools.custom_tool import TranscriptionTool, SceneChangeTimeStamp, FrameScore, SceneScoreingTool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class ContentCrew:
    """Content Crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def trasncriber(self) -> Agent:
        return Agent(
            config=self.agents_config["transcriber"],  # type: ignore[index]
        )

    @task
    def transcription_task(self) -> Task:
        return Task(
            tools=[TranscriptionTool],
            config=self.tasks_config["transcription_task"],  # type: ignore[index]
        )
    
    @agent
    def transcript_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["transcript_analyst"], # type: ignore[index]
        )
    
    @task
    def transcript_analysis_task(self) -> Task:
        return Task(
            config= self.tasks_config["transcript_analysis_task"], # type: ignore[index]
        )

    @agent
    def content_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["content_planner"], # type: ignore[index]
        )
    
    @task
    def content_planning_task(self) -> Task:
        return Task(
            config=self.tasks_config["content_planning_task"], # type: ignore[index]
            output_pydantic=list[SceneChangeTimeStamp]
        )
    
    @agent
    def scene_scorer(self) -> Agent:
        return Agent(
            config=self.agents_config["scene_scorer"], # type: ignore[index]
        )

    @task
    def scene_scoring_task(self) -> Task:
        return Task(
            config = self.tasks_config["scene_scoring_task"], # type: ignore[index]
            tools = [SceneScoreingTool],
            output_pydantic=list[FrameScore]
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Content Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
        )
