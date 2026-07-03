#!/usr/bin/env python
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
    parts: list[list[dict]] = []

class ContentFlow(Flow[ContentState]):

    @start()
    def plan_content(self):
        self.state.video = "/Users/helen/Desktop/ai-automation-portfolio/Part 2/testing/[TubePull] Im_scared_of_my_own_autistic_child_-_BBC_News.mp4"

    @listen(plan_content)
    def detect_scenes(self):
        self.state.scene_timestamps = SceneChangeDetectionTool().detect_timestamps(self.state.video)

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
        if self.state.strategy == "audio":
            return
        scene_timestamps_with_frames = SceneChangeDetectionTool().extract_frames(self.state.video)
        scores = SceneScoringTool()._run([s.model_dump() for s in scene_timestamps_with_frames])
        self.state.scored_scenes = [s.model_dump() for s in scores]

    @listen(score_scenes)
    def merge_timeline(self):
        if self.state.strategy == "audio":
            self.state.master_timeline = [
                {
                    "start_time": t["start_time"],
                    "end_time": t["end_time"],
                    "transcription": t["transcription"],
                    "motion_intensity": 0,
                    "emotional_intensity": 0,
                    "composition_quality": 0,
                    "scene_description": "",
                }
                for t in self.state.transcriptions
            ]
            return
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
        self.state.parts = [
            [t.model_dump() for t in part.timestamps]
            for part in result.pydantic.parts  # type: ignore[union-attr]
        ] if result.pydantic else []
    
    @listen(generate_content_plan)
    def execute_edit(self):
        sentence_ends = [
            t["end_time"]
            for t in self.state.transcriptions
            if t["transcription"].strip().endswith((".", "?", "!"))
        ]

        def snap(end_time: float, window: float = 8.0) -> float:
            nearby = [t for t in sentence_ends if abs(t - end_time) <= window]
            if not nearby:
                return end_time
            after = [t for t in nearby if t >= end_time]
            return min(after) if after else max(nearby)

        def process_part(timestamps: list[dict]) -> list[dict]:
            snapped = sorted(
                [{"start_time": ts["start_time"], "end_time": snap(ts["end_time"])} for ts in timestamps],
                key=lambda x: x["start_time"]
            )

            deduped: list[dict] = []
            for seg in snapped:
                if deduped and seg["start_time"] < deduped[-1]["end_time"]:
                    deduped[-1]["end_time"] = max(deduped[-1]["end_time"], seg["end_time"])
                else:
                    deduped.append(seg)

            split: list[dict] = []
            for seg in deduped:
                if seg["end_time"] - seg["start_time"] <= 60:
                    split.append(seg)
                    continue
                boundaries = sorted([s for s in sentence_ends if seg["start_time"] < s < seg["end_time"]])
                if not boundaries:
                    split.append(seg)
                    continue
                all_points = [seg["start_time"]] + boundaries + [seg["end_time"]]
                chunk_start_idx = 0
                for i in range(1, len(all_points)):
                    if all_points[i] - all_points[chunk_start_idx] > 60:
                        if i - 1 > chunk_start_idx:
                            split.append({"start_time": all_points[chunk_start_idx], "end_time": all_points[i - 1]})
                            chunk_start_idx = i - 1
                        else:
                            split.append({"start_time": all_points[chunk_start_idx], "end_time": all_points[i]})
                            chunk_start_idx = i
                if chunk_start_idx < len(all_points) - 1:
                    split.append({"start_time": all_points[chunk_start_idx], "end_time": all_points[-1]})

            final: list[dict] = []
            for seg in split:
                if seg["end_time"] - seg["start_time"] < 1.0:
                    if final:
                        final[-1]["end_time"] = seg["end_time"]
                    continue
                final.append(seg)

            return final

        for i, part in enumerate(self.state.parts):
            segments = process_part(part)
            if not segments:
                continue
            output = VideoEditingTool()._run(
                video_file_path=self.state.video,
                timestamps=segments,
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
