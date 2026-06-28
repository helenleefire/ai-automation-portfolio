#!/usr/bin/env python
from pathlib import Path
import os

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from video_editing_flow.crews.content_crew.content_crew import ContentCrew
from video_editing_flow.tools.custom_tool import SceneChangeDetectionTool, VideoEditingTool, AudioChunkingTool, TranscriptionTool


class ContentState(BaseModel):
    video: str = ""
    scene_timestamps: list[dict] = []
    transcriptions: list[dict] = []
    timestamps: list[dict] = []
    final_post: str = ""

class ContentFlow(Flow[ContentState]):

    @start()
    def plan_content(self):
        self.state.video = "/Users/helen/Desktop/ai-automation-portfolio/Part 2/testing/video.mp4"

    @listen(plan_content)
    def detect_scenes(self):
        scenes = SceneChangeDetectionTool()._run(self.state.video)
        self.state.scene_timestamps = [s.model_dump() for s in scenes]

    @listen(detect_scenes)
    def transcribe(self):
        chunks = AudioChunkingTool()._run(
            video_file_path=self.state.video,
            scene_timestamps=self.state.scene_timestamps
        )
        transcriptions = TranscriptionTool()._run(chunks)
        self.state.transcriptions = [t.model_dump() for t in transcriptions]

    @listen(transcribe)
    def generate_content_plan(self):
        result = (
            ContentCrew()
            .crew()
            .kickoff(inputs={
                "video": self.state.video,
                "scene_timestamps": self.state.scene_timestamps,
                "transcriptions": self.state.transcriptions,
            })
        )
        self.state.timestamps = [t.model_dump() for t in result.pydantic.timestamps] if result.pydantic else []  # type: ignore[union-attr]
    
    @listen(generate_content_plan)
    def execute_edit(self):
        output = VideoEditingTool()._run(
            video_file_path=self.state.video,
            timestamps=self.state.timestamps,
            output_path=os.path.abspath("final_output.mp4")
        )
        print(f"Final video saved to {output}")


def kickoff():
    content_flow = ContentFlow()
    content_flow.kickoff()


def plot():
    content_flow = ContentFlow()
    content_flow.plot()


def run_with_trigger():
    """
    Run the flow with trigger payload.
    """
    import json
    import sys

    # Get trigger payload from command line argument
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    # Create flow and kickoff with trigger payload
    # The @start() methods will automatically receive crewai_trigger_payload parameter
    content_flow = ContentFlow()

    try:
        result = content_flow.kickoff({"crewai_trigger_payload": trigger_payload})
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the flow with trigger: {e}")


if __name__ == "__main__":
    kickoff()
