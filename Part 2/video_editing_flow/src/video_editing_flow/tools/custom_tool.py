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

class ScoredScenes(BaseModel):
    """Schema to be used by scene_scoring_task"""
    scenes: list[FrameScore] = Field(..., description="list of scored scenes with timestamps")

class ContentPlan(BaseModel):
    """Schema to be used by content_planning_task"""
    timestamps: list[TimeStamp] = Field(..., description="list of start/end timestamps of segments to include in the final video")

class SceneScoringTool(BaseTool):
    name: str = "scene scoring tool"
    description: str = "This tool should be used to score scenes from looking at still representing them"
    
    def _run(self, scene_change_time_stamps: list[SceneChangeTimeStamp]) -> list[FrameScore]:
        scene_change_time_stamps = [SceneChangeTimeStamp(**s) if isinstance(s, dict) else s for s in scene_change_time_stamps]
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
                max_tokens=2048,
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
        frames_dir = os.path.abspath("frames")
        os.makedirs(frames_dir, exist_ok=True)

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
            os.path.join(frames_dir, "frame_%04d.jpg")
        ], capture_output=True, text=True)

        timestamps = [float(t) for t in re.findall(r"pts_time:([\d.]+)", result.stderr)]

        if timestamps and timestamps[0] > 0:
            first_frame = os.path.join(frames_dir, "frame_0000.jpg")
            subprocess.run([
                "ffmpeg", "-i", video_file_path,
                "-ss", "0", "-vframes", "1",
                first_frame
            ], capture_output=True)
            timestamps = [0.0] + timestamps

        scenes = []
        for i, start in enumerate(timestamps):
            end = timestamps[i + 1] if i + 1 < len(timestamps) else duration
            scenes.append(SceneChangeTimeStamp(
                start_time=start,
                end_time=end,
                frame_file_path=os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            ))

        return scenes

class TranscriptionTool(BaseTool):
    name: str = "Transcription tool"
    description: str = "Transcribes a video file and returns segments with timestamps."

    def _run(self, video_file_path: str) -> list[TranscriptionWithTimeStamp]:
        return self.transcribe_video(video_file_path)

    def transcribe_video(self, video_file_path: str) -> list[TranscriptionWithTimeStamp]:
        audio_path = os.path.abspath("full_audio.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_file_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
            audio_path
        ], capture_output=True)

        model = WhisperModel("base", device="cpu", compute_type="float32")
        segments, _ = model.transcribe(audio_path)

        return [
            TranscriptionWithTimeStamp(
                start_time=segment.start,
                end_time=segment.end,
                transcription=segment.text
            )
            for segment in segments
        ]


class MasterTimeStamp(FrameScore):
    """Schema to be used by ContentPlanningTool"""
    transcription: str = Field(..., description="raw text value of transcription")

class MasterTimeline(BaseModel):
    """Schema to be used by timeline_merge_task"""
    timeline: list[MasterTimeStamp] = Field(..., description="merged timeline of scored scenes with transcriptions")

class TimelineMergeToolInput(BaseModel):
    frame_scores: list[FrameScore] = Field(..., description="scored frames with timestamps")
    transcriptions: list[TranscriptionWithTimeStamp] = Field(..., description="transcriptions with timestamps")

class TimelineMergeTool(BaseTool):
    name: str = "Tool to merge transcription and scene time stamps"
    description: str = "This tool should be used to create the master time line to be used by content planner agent"
    args_schema: Type[BaseModel] = TimelineMergeToolInput

    def _run(self, frame_scores: list[FrameScore], transcriptions: list[TranscriptionWithTimeStamp], strategy: str = "visual") -> list[MasterTimeStamp]:
        frame_scores = [FrameScore(**f) if isinstance(f, dict) else f for f in frame_scores]
        transcriptions = [TranscriptionWithTimeStamp(**t) if isinstance(t, dict) else t for t in transcriptions]

        if strategy == "audio":
            return self._merge_by_sentence(frame_scores, transcriptions)
        return self._merge_by_scene(frame_scores, transcriptions)

    def _merge_by_scene(self, frame_scores: list[FrameScore], transcriptions: list[TranscriptionWithTimeStamp]) -> list[MasterTimeStamp]:
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

    def _merge_by_sentence(self, frame_scores: list[FrameScore], transcriptions: list[TranscriptionWithTimeStamp]) -> list[MasterTimeStamp]:
        result = []
        for t in transcriptions:
            overlapping = [f for f in frame_scores if f.start_time < t.end_time and f.end_time > t.start_time]
            best = (
                max(overlapping, key=lambda f: min(f.end_time, t.end_time) - max(f.start_time, t.start_time))
                if overlapping
                else min(frame_scores, key=lambda f: abs(f.start_time - t.start_time))
            )
            result.append(MasterTimeStamp(
                start_time=t.start_time,
                end_time=t.end_time,
                motion_intensity=best.motion_intensity,
                emotional_intensity=best.emotional_intensity,
                composition_quality=best.composition_quality,
                scene_description=best.scene_description,
                transcription=t.transcription,
            ))
        return result


class VideoEditingToolInput(BaseModel):
    video_file_path: str = Field(..., description="path to the source video file")
    timestamps: list[TimeStamp] = Field(..., description="list of start/end timestamps to cut and stitch")
    output_path: str = Field(..., description="path for the final output video")

class VideoEditingTool(BaseTool):
    name: str = "Video editing tool"
    description: str = "Cuts the source video at the given timestamps and stitches the segments into a final output video"
    args_schema: Type[BaseModel] = VideoEditingToolInput

    def _run(self, video_file_path: str, timestamps: list[TimeStamp], output_path: str) -> str:
        timestamps = [TimeStamp(**t) if isinstance(t, dict) else t for t in timestamps]

        segments_dir = os.path.abspath("segments")
        os.makedirs(segments_dir, exist_ok=True)

        segment_paths = []
        for i, ts in enumerate(timestamps):
            out = os.path.join(segments_dir, f"segment_{i+1:04d}.mp4")
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(ts.start_time),
                "-i", video_file_path,
                "-t", str(ts.end_time - ts.start_time),
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                out
            ], capture_output=True)
            segment_paths.append(out)

        filelist = os.path.join(segments_dir, "filelist.txt")
        with open(filelist, "w") as f:
            for path in segment_paths:
                f.write(f"file '{path}'\n")

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", filelist,
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            output_path
        ], capture_output=True)

        return output_path
