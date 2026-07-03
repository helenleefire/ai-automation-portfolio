#!/usr/bin/env python
import math
import os
import shutil

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from video_editing_flow.crews.content_crew.content_crew import ContentCrew
from video_editing_flow.tools.custom_tool import SceneChangeDetectionTool, VideoEditingTool, TranscriptionTool, SceneScoringTool, TimelineMergeTool


class ContentState(BaseModel):
    video: str = ""
    scene_timestamps: list[dict] = []
    transcriptions: list[dict] = []
    strategy: str = "visual"
    scored_scenes: list[dict] = []
    master_timeline: list[dict] = []
    timestamps: list[dict] = []

class ContentFlow(Flow[ContentState]):

    @start()
    def plan_content(self):
        self.state.video = "/Users/helen/Desktop/ai-automation-portfolio/Part 2/testing/[TubePull] Silicon_Valley_Moved_to_Austin_Then_Regretted_It.mp4"

    @listen(plan_content)
    def detect_scenes(self):
        scenes = SceneChangeDetectionTool()._run(self.state.video)
        self.state.scene_timestamps = [s.model_dump() for s in scenes]

    @listen(detect_scenes)
    def transcribe(self):
        transcriptions = TranscriptionTool().transcribe_video(self.state.video)
        self.state.transcriptions = [t.model_dump() for t in transcriptions]

    @listen(transcribe)
    def detect_strategy(self):
        total_duration = self.state.scene_timestamps[-1]["end_time"] if self.state.scene_timestamps else 1
        speech_duration = sum(t["end_time"] - t["start_time"] for t in self.state.transcriptions)
        self.state.strategy = "audio" if speech_duration / total_duration > 0.6 else "visual"
        print(f"Strategy: {self.state.strategy} (speech ratio: {speech_duration / total_duration:.0%})")

    @listen(detect_strategy)
    def score_scenes(self):
        scores = SceneScoringTool()._run(self.state.scene_timestamps)
        self.state.scored_scenes = [s.model_dump() for s in scores]

    @listen(score_scenes)
    def merge_timeline(self):
        merged = TimelineMergeTool()._run(
            frame_scores=self.state.scored_scenes,
            transcriptions=self.state.transcriptions,
            strategy=self.state.strategy,
        )
        self.state.master_timeline = [m.model_dump() for m in merged]

    @listen(merge_timeline)
    def generate_content_plan(self):
        result = (
            ContentCrew()
            .crew(self.state.strategy)
            .kickoff(inputs={
                "master_timeline": self.state.master_timeline,
            })
        )
        self.state.timestamps = [t.model_dump() for t in result.pydantic.timestamps] if result.pydantic else []  # type: ignore[union-attr]
    
    @listen(generate_content_plan)
    def execute_edit(self):
        transcriptions = self.state.transcriptions
        sentence_ends = [
            t["end_time"]
            for t in transcriptions
            if t["transcription"].strip().endswith((".", "?", "!"))
        ]

        def snap(end_time: float) -> float:
            after = [t for t in sentence_ends if t >= end_time]
            before = [t for t in sentence_ends if t < end_time]
            if after:
                return min(after)
            return max(before) if before else end_time

        snapped = sorted(
            [{"start_time": ts["start_time"], "end_time": snap(ts["end_time"])} for ts in self.state.timestamps],
            key=lambda x: x["start_time"]
        )

        deduped = []
        for seg in snapped:
            if deduped and seg["start_time"] < deduped[-1]["end_time"]:
                deduped[-1]["end_time"] = max(deduped[-1]["end_time"], seg["end_time"])
            else:
                deduped.append(seg)

        total_duration = sum(s["end_time"] - s["start_time"] for s in deduped)

        num_videos = 1
        current = 0.0
        for seg in deduped:
            d = seg["end_time"] - seg["start_time"]
            if current > 0 and current + d > 60:
                num_videos += 1
                current = d
            else:
                current += d

        target = total_duration / num_videos

        groups: list[list[dict]] = []
        current_group: list[dict] = []
        current_duration = 0.0

        for seg in deduped:
            seg_duration = seg["end_time"] - seg["start_time"]
            would_exceed = current_duration + seg_duration > 60
            past_target = abs(current_duration + seg_duration - target) > abs(current_duration - target)
            if (current_group
                    and len(groups) < num_videos - 1
                    and (would_exceed or past_target)):
                groups.append(current_group)
                current_group = [seg]
                current_duration = seg_duration
            else:
                current_group.append(seg)
                current_duration += seg_duration

        if current_group:
            groups.append(current_group)

        for i, group in enumerate(groups):
            output = VideoEditingTool()._run(
                video_file_path=self.state.video,
                timestamps=group,
                output_path=os.path.abspath(f"final_output_{i + 1}.mp4")
            )
            print(f"Video {i + 1} saved to {output}")

    @listen(execute_edit)
    def cleanup(self):
        audio = os.path.abspath("full_audio.wav")
        if os.path.exists(audio):
            os.remove(audio)
        for folder in ["frames", "segments"]:
            path = os.path.abspath(folder)
            if os.path.exists(path):
                shutil.rmtree(path)


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
