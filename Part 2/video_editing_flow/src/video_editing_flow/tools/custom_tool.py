from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from faster_whisper import WhisperModel
import anthropic, instructor, base64, subprocess, os, re

class TimeStamp(BaseModel):
    start_time: float = Field(..., description="start time of a segment")
    end_time: float = Field(..., description="end time of a segment")

class FrameScore(BaseModel):
    """Schema to be used by FrameScoreTool"""
    scene_number: int = Field(..., description="scene number")
    motion_intensity: int = Field(..., description="intensity of motion scored from 1 to 5 compared to other stills")
    emotional_intensity: int = Field(..., description="intensity of emotion conveyed scored from 1 to 5 compared to other stills")
    composition_quality: int = Field(..., description="composition quality scored from 1 to 5 compared to other stills")
    scene_description: str = Field(..., description="description of what is happening on still to be referenced along with various frame scores")

class SceneChangeTimeStamp(TimeStamp):
    """Schema to be used by ContentPlanningTool"""
    scene_number: int = Field(..., description="scene number")
    frame_file_path: str = Field(..., description="file path of still representing scene")

class TranscriptionWithTimeStamp(TimeStamp):
    """Schema to be used by TranscriptionTool"""
    transcription: str = Field(..., description="raw text value of transcription")

class SceneScoreingTool(BaseTool):
    name: str = "Scene scoring tool"
    description: str = "This tool should be used to score scenes from looking at still representing them"
    def _run(self, scene_change_time_stamps: list[SceneChangeTimeStamp]) -> list[FrameScore]:
        content =[]
        for scene in scene_change_time_stamps:
            with open(scene.frame_file_path, "rb") as f:
                img = base64.b64encode(f.read()).decode()
            content.append({"type": "text", "text": str(scene.scene_number)})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img}
            })
        content.append({"type": "text", "text": "Score each frames on given qualities against each other"})
        client = instructor.from_anthropic(anthropic.Anthropic())

        response = client.messages.create(
            max_tokens= 500,
            model="claude-sonnet-4-6",
            response_model = list[FrameScore],
            messages= [
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        return response

class SceneChangeDetectionTool(BaseTool):
    name: str = "Scene change detection tool"
    description: str = "This tool should be used to extract all of the scene changes which will be used to find the most visually significant frames"
    def _run(self, video_file_path: str) -> list[SceneChangeTimeStamp]:
        os.makedirs("frames", exist_ok=True)

        duration_result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            video_file_path
        ], capture_output=True, text=True)
        duration = float(duration_result.stdout.strip())

        result = subprocess.run([ 
            "ffmpeg", "-i", video_file_path,
            "-vf", "select=gt(scene\\,0.3),showinfo",
            "-vsync", "vfr",
            "frames/frame_%04d.jpg"
        ], capture_output=True, text=True)
    
        timestamps = [float(t) for t in re.findall(r"pts_time:([\d.]+)", result.stderr)]

        scenes = []
        for i, start in enumerate(timestamps):
            end = timestamps[i + 1] if i + 1 < len(timestamps) else duration
            scenes.append(SceneChangeTimeStamp(
                start_time=start,
                end_time=end,
                frame_file_path=f"frames/frame_{i + 1:04d}.jpg",
                scene_number=i + 1
            ))

        return scenes

class TranscriptionTool(BaseTool):
    name: str = "Transcription tool"
    description: str = "This tool should be used to extract the raw string value of the transcription of a video accompanied by start and end time stamps"
    def _run(self, video_file_path: str) -> list[TimeStamp]:
        model = WhisperModel("base", device="cpu")
        segments, info = model.transcribe(video_file_path)
        transcription = []
        for segment in segments:
            transcription.append(TranscriptionWithTimeStamp(start_time=segment.start, end_time=segment.end, transcription=segment.text))
        return transcription
