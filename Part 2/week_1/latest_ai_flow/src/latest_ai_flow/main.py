from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from latest_ai_flow.crews.content_crew.content_crew import ResearchCrew


class ResearchFlowState(BaseModel):
  topic: str = ""
  report: str = ""
  summary: str= ""


class LatestAiFlow(Flow[ResearchFlowState]):
  @start()
  def prepare_topic(self, crewai_trigger_payload: dict | None = None):
    if crewai_trigger_payload:
      self.state.topic = crewai_trigger_payload.get("topic", "AI Agents")
    else:
      self.state.topic = "AI Agents"
    print(f"Topic: {self.state.topic}")

  @listen(prepare_topic)
  def run_research(self):
    result = ResearchCrew().crew().kickoff(inputs={"topic": self.state.topic})
    self.state.report = result.raw
    print("Research crew finished.")

  @listen(run_research)
  def report(self):
    print("Report path: output/report.md")
    print("Summary path: output/summary.md")
    
def kickoff():
  LatestAiFlow().kickoff()


def plot():
  LatestAiFlow().plot()


if __name__ == "__main__":
  kickoff()