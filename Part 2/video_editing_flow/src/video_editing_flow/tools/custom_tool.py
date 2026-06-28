from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from faster_whisper import WhisperModel
from typing import Type
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

class AudioChunk(BaseModel):
    start_time: float = Field(..., description="start time of audio chunk in the video")
    end_time: float = Field(..., description="end time of audio chunk in the video")
    file_path: str = Field(..., description="file path of extracted audio chunk")

class AudioChunkingToolInput(BaseModel):
    video_file_path: str = Field(..., description="path to video file")
    scene_timestamps: list[SceneChangeTimeStamp] = Field(..., description="scene change timestamps")

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

class AudioChunkingTool(BaseTool):
    name: str = "Tool to extract audio from video and chunk them referencing the scene timeline"
    description: str = "This tool will be used before transcription to help with merging of transcription timeline with scene change timeline"
    args_schema: Type[BaseModel] = AudioChunkingToolInput

    def _run(self, video_file_path: str, scene_timestamps: list[SceneChangeTimeStamp]) -> list[AudioChunk]:
        os.makedirs("audio_chunks", exist_ok=True)

        chunks = []
        current_start = scene_timestamps[0].start_time
        current_end = scene_timestamps[0].end_time

        for scene in scene_timestamps[1:]:
            if current_end - current_start < 5:
                current_end = scene.end_time
            else:
                chunks.append((current_start, current_end))
                current_start = scene.start_time
                current_end = scene.end_time

        chunks.append((current_start, current_end))

        audio_chunks = []
        for i, (start, end) in enumerate(chunks):
            file_path = f"audio_chunks/chunk_{i + 1:04d}.wav"
            subprocess.run([
                "ffmpeg", "-i", video_file_path,
                "-ss", str(start),
                "-to", str(end),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
                file_path
            ], capture_output=True)
            audio_chunks.append(AudioChunk(start_time=start, end_time=end, file_path=file_path))

        return audio_chunks

class TranscriptionTool(BaseTool):
    name: str = "Transcription tool"
    description: str = "This tool should be used to extract the raw string value of the transcription of a video accompanied by start and end time stamps. This should be done considering scene change time stamps for boundaries."

    def _run(self, audio_chunks: list[AudioChunk]) -> list[TranscriptionWithTimeStamp]:
        model = WhisperModel("base", device="cpu")
        transcriptions = []

        for chunk in audio_chunks:
            segments, _ = model.transcribe(chunk.file_path)
            for segment in segments:
                transcriptions.append(TranscriptionWithTimeStamp(
                    start_time=chunk.start_time + segment.start,
                    end_time=chunk.start_time + segment.end,
                    transcription=segment.text
                ))

        return transcriptions

class TimelineMergeToolInput(BaseModel):
    frame_scores: list[FrameScore] = Field(..., description="scored frames with timestamps")
    transcriptions: list[TranscriptionWithTimeStamp] = Field(..., description="transcriptions with timestamps")

class TimelineMergeTool(BaseTool):
    name: str = "Tool to merge transcription and scene time stamps"
    description: str = "This tool should be used to create the master time line to be used by content planner agent"
    args_schema: Type[BaseModel] = TimelineMergeToolInput

    def _run(self, frame_scores: list[FrameScore], transcriptions: list[TranscriptionWithTimeStamp]) -> list[MasterTimeStamp]:
        master_timeline = []

        for frame in frame_scores:
            overlapping = [
                t.transcription for t in transcriptions
                if t.start_time < frame.end_time and t.end_time > frame.start_time
            ]
            master_timeline.append(MasterTimeStamp(
                start_time=frame.start_time,
                end_time=frame.end_time,
                motion_intensity=frame.motion_intensity,
                emotional_intensity=frame.emotional_intensity,
                composition_quality=frame.composition_quality,
                scene_description=frame.scene_description,
                transcription=" ".join(overlapping)
            ))

        return master_timeline
