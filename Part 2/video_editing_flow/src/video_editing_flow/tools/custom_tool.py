from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from faster_whisper import WhisperModel
import asyncio, anthropic, instructor, base64, subprocess, os, re

class TimeStamp(BaseModel):
    start_time: float = Field(..., description="start time of a scene")
    end_time: float = Field(..., description="end time of a scene")

class SceneChangeTimeStamp(TimeStamp):
    """Schema to be used by SceneScoreingTool"""
    frame_file_path: str = Field(..., description="file path of still representing scene")

class TranscriptionWithTimeStamp(TimeStamp):
    """Schema to be used by TranscriptionTool"""
    transcription: str = Field(..., description="raw text value of transcription")

class FrameScore(TimeStamp):
    """Schema to be used by FrameScoreTool"""
    motion_intensity: int = Field(..., description="intensity of motion scored from 1 to 5 compared to other stills")
    emotional_intensity: int = Field(..., description="intensity of emotion conveyed scored from 1 to 5 compared to other stills")
    composition_quality: int = Field(..., description="composition quality scored from 1 to 5 compared to other stills")
    scene_description: str = Field(..., description="description of what is happening on still to be referenced along with various frame scores")

class MasterTimeStamp(FrameScore):
    """Schema to be used by ContentPlanningTool"""
    transcription: str = Field(..., description="raw text value of transcription")

class SceneScoringTool(BaseTool):
    name: str = "scene scoring tool"
    description: str = "This tool should be used to score scenes from looking at still representing them"
    
    def _run(self, scene_change_time_stamps: list[SceneChangeTimeStamp]) -> list[FrameScore]:
        return asyncio.run(self._arun(scene_change_time_stamps))
    
    async def _arun(self, scene_change_time_stamps: list[SceneChangeTimeStamp]) -> list[FrameScore]:
        sem = asyncio.Semaphore(3)
        chunks = [
            scene_change_time_stamps[i:i + 5]
            for i in range(0, len(scene_change_time_stamps), 5)
        ]
        results = await asyncio.gather(*[self._score_chunk(chunk, sem) for chunk in chunks]) 
        return [score for chunk_scores in results for score in chunk_scores]
    
    async def _score_chunk(self, chunk: list[SceneChangeTimeStamp], sem: asyncio.Semaphore) -> list[FrameScore]:
        async with sem:
            content = []
            for scene in chunk:
                with open(scene.frame_file_path, "rb") as f:
                    img = base64.b64encode(f.read()).decode()
                content.append({"type": "text", "text": str(scene.start_time)})
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
    name: str = "scene change detection tool"
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
                frame_file_path=f"frames/frame_{i + 1:04d}.jpg"
            ))

        return scenes

# finish writing this function
class AudioChunkingTool(BaseTool):
    name: str = "Tool to extract audio from video and chunk them referencing the scene timeline"
    description: str = "This tool will be used before transcription to help with merging of transcription timeline with scene change timeline"

# rewrite this so it accepts list of audio files and create unified timeline
class TranscriptionTool(BaseTool):
    name: str = "Transcription tool"
    description: str = "This tool should be used to extract the raw string value of the transcription of a video accompanied by start and end time stamps. This should be done considering scene change time stamps for boundaries."
    def _run(self, video_file_path: str) -> list[TranscriptionWithTimeStamp]:
        model = WhisperModel("base", device="cpu")
        scenes, info = model.transcribe(video_file_path)
        transcription = []
        for scene in scenes:
            transcription.append(TranscriptionWithTimeStamp(start_time=scene.start, end_time=scene.end, transcription=scene.text))
        return transcription

# finish writing this function
class TimelineMergeTool(BaseTool):
    name: str = "Tool to merge transcription and scene time stamps"
    description: str = "This tool should be used to create the master time line to be used by content planner agent"
