from typing import List

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool


@CrewBase
class ResearchCrew:
  """Single-agent research crew used inside the Flow."""

  agents: List[BaseAgent]
  tasks: List[Task]

  agents_config = "config/agents.yaml"
  tasks_config = "config/tasks.yaml"

  @agent
  def researcher(self) -> Agent:
    return Agent(
      config=self.agents_config["researcher"],  # type: ignore[index]
      verbose=True,
      tools=[SerperDevTool()],
      llm=LLM(model="anthropic/claude-sonnet-4-6"),
    )

  @task
  def research_task(self) -> Task:
    return Task(
      config=self.tasks_config["research_task"],  # type: ignore[index]
    )
  
  @agent
  def summarizer(self)-> Agent:
    return Agent(
      config=self.agents_config["summarizer"], # type: ignore[index]
      verbose=False,
      llm=LLM(model="anthropic/claude-sonnet-4-6"),
    )

  @task
  def summarize_task(self) -> Task:
    return Task(
      config=self.tasks_config["summarize_task"], # type: ignore[index]
    )

  @crew
  def crew(self) -> Crew:
    return Crew(
      agents=self.agents,
      tasks=self.tasks,
      process=Process.sequential,
      verbose=True,
    )