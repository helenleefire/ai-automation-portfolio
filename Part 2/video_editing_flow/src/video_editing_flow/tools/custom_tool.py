from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from faster_whisper import WhisperModel
import subprocess, re
    
class TimeStamp(BaseModel):
    """Schema to be used by TranscriptionTool and ContentPlanningTool"""
    start_time: float = Field(..., description="start time of a segment")
    end_time: float = Field(..., description="end time of a segment")
    transcription: str = Field(..., description="raw text value of transcription")
    # scene_description: str = Field(..., description="description of what is happening visually")

class SceneChangeDetectionTool(BaseTool):
    name: str = "Scene change detection tool"
    description: str = "This tool should be used to extract all of the scene changes which will be used to find the most visually significant frames"
    def _run(self, video_file_path: str) -> list[TimeStamp]:
        duration_result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            video_file_path
        ], capture_output=True, text=True)
        duration = float(duration_result.stdout.strip())
        scenes = []
        scenes.append(TimeStamp(start_time=0, end_time=duration, transcription=""))
        for i in (1, len(scenes)):
            scenes[i].start_time = scenes[i - 1].end_time
        return scenes

class TranscriptionTool(BaseTool):
    name: str = "Transcription tool"
    description: str = "This tool should be used to extract the raw string value of the transcription of a video accompanied by start and end time stamps"
    def _run(self, video_file_path: str) -> list[TimeStamp]:
        model = WhisperModel("base", device="cpu")
        segments, info = model.transcribe(video_file_path)
        transcription = []
        for segment in segments:
            transcription.append(TimeStamp(start_time=segment.start, end_time=segment.end, transcription=segment.text))
        return transcription
