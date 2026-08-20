# AI & Automation Portfolio

Multi-agent LLM systems built with CrewAI and LangChain, plus the exercises I worked through while building toward them.

---

## Automated Video Editing Pipeline

`Part 2/video_editing_flow/`

A CrewAI Flow that turns long-form video into short-form clips by merging visual and audio analysis into a single scored timeline.

**Pipeline**

1. **Scene detection** — FFmpeg scene-change filter produces cut candidates
2. **Transcription** — Whisper generates timestamped speech segments
3. **Strategy selection** — computes the speech-to-duration ratio and routes to an `audio` or `visual` editing strategy
4. **Frame scoring** — extracted frames are base64-encoded and scored by Claude vision on motion intensity, emotional intensity, and composition quality
5. **Timeline merge** — visual scores and transcript segments merge into one master timeline, by scene or by sentence depending on the selected strategy
6. **Content planning** — a CrewAI crew selects clip ranges from the merged timeline
7. **Render** — cut points snap to the nearest sentence boundary within an 8-second window, then FFmpeg extracts the final parts

**Stack** — CrewAI Flows, Whisper, Claude vision, FFmpeg, Pydantic

**Key files** — [`main.py`](Part%202/video_editing_flow/src/video_editing_flow/main.py) for the flow state machine, [`custom_tool.py`](Part%202/video_editing_flow/src/video_editing_flow/tools/custom_tool.py) for the five tools

---

## Screenwriting Agent

`Part 1/screenwriting_helper/`

A multi-tool LangChain agent for screenwriters that generates scene outlines conversationally and analyzes uploaded scripts.

**Design**

- **Scene-aware chunking** splits scripts on `INT.` and `EXT.` slug lines so retrieval returns whole scenes rather than arbitrary character spans
- **Two separate Chroma stores** keep a reference corpus of published screenplays apart from the user's own script, so analysis and inspiration never bleed together
- **MMR retrieval** (`k=5`, `fetch_k=20`) with HuggingFace `all-mpnet-base-v2` embeddings favors diverse passages over near-duplicates
- **Analysis tools** evaluate pacing, character consistency, and three-act structure

**Stack** — LangChain, Chroma, HuggingFace embeddings, Claude

**Key files** — [`agent_tools.py`](Part%201/screenwriting_helper/agent_tools.py), [`ingest_data.py`](Part%201/screenwriting_helper/ingest_data.py), [`screenwriting_agent.py`](Part%201/screenwriting_helper/screenwriting_agent.py)

---

## Multi-Tool Support Agent

`Part 1/week_7/`

A LangChain agent for employee and IT support that routes natural-language queries across document retrieval, record lookup, ticket creation, classification, and escalation tools. Includes an evaluation harness in [`eval.py`](Part%201/week_7/eval.py).

**Stack** — LangChain, RAG, Python

---

## Foundations

`Part 1/week_1` through `week_7` trace the build-up to the projects above: router and summarizer agents, document loaders and retrievers, prompt experiments, structured output with Pydantic, LlamaIndex, and agent evaluation.
