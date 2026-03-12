# AI Video Q&A System — Implementation Guide

A detailed, step-by-step guide to build an AI system where users can upload a video, ask natural language questions about the content, and receive accurate answers (e.g., *"Tell me the color and model of the red car"* or *"Give me a summary of this video"*).

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Tech Stack](#2-tech-stack)
3. [AI Models](#3-ai-models)
4. [Workflow Design](#4-workflow-design)
5. [Code Structure](#5-code-structure)
6. [Implementation Steps](#6-implementation-steps)

---

## 1. High-Level Architecture

### System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              USER / FRONTEND                                              │
│  [Upload Video]  ──────────────────────────►  [Ask Question: "What color is the car?"]   │
└────────────────────────┬──────────────────────────────────────────┬──────────────────────┘
                         │                                          │
                         ▼                                          │
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         NODE.JS (EXPRESS) — API Gateway                                 │
│  • Port: 5000                                                                           │
│  • Handles: Auth, User Management, Video Upload, Request Routing                        │
│  • Receives: Multipart video upload + user prompt/question                              │
└────────────────────────┬───────────────────────────────────────────────────────────────┘
                         │
                         │ 1. Save video to uploads/
                         │ 2. POST to Python: { videoPath, question }
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       PYTHON (FASTAPI) — AI Microservice                                │
│  • Port: 8000                                                                           │
│  • Handles: Video processing, Frame extraction, Audio transcription, AI inference       │
└────────────────────────┬───────────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐
│   FFmpeg    │  │   Whisper   │  │  Multimodal Model   │
│ Extract     │  │ Transcribe  │  │  (GPT-4V / Gemini)  │
│ Frames      │  │ Audio       │  │  Answer question    │
└──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘
       │                │                    │
       └────────────────┼────────────────────┘
                        │
                        │  Combined prompt:
                        │  [Frames + Transcript] + User Question
                        │
                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         RESPONSE                                                        │
│  Node.js receives answer from Python → returns to frontend                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Step | Component | Action |
|------|-----------|--------|
| 1 | Frontend | User uploads video + submits question |
| 2 | Node.js | Validates user, saves video, forwards to Python |
| 3 | Python | Extracts frames (FFmpeg/OpenCV), transcribes audio (Whisper) |
| 4 | Python | Sends frames + transcript + question to multimodal AI |
| 5 | Multimodal AI | Returns answer |
| 6 | Python | Returns answer to Node.js |
| 7 | Node.js | Returns answer to user |

---

## 2. Tech Stack

### Backend: Python (FastAPI) + Node.js (Express)

| Layer | Technology | Role |
|-------|------------|------|
| **API Gateway** | Node.js + Express | Auth, user management, upload handling, request routing |
| **AI Microservice** | Python + FastAPI | Video processing, frame extraction, Whisper, multimodal AI calls |

### Why Use Both Together?

| Benefit | Python (FastAPI) | Node.js (Express) |
|---------|------------------|-------------------|
| **AI/ML ecosystem** | Native support: PyTorch, OpenCV, Whisper, FFmpeg bindings | Limited ML libraries |
| **Async I/O** | Good for CPU-bound tasks, async when needed | Excellent for I/O-bound: DB, file uploads, external APIs |
| **User management** | Can do it, but JS ecosystem is stronger for web APIs | JWT, bcrypt, Mongoose, middleware patterns |
| **Video processing** | FFmpeg, OpenCV, subprocess calls — natural fit | Would require child_process + Python subprocess anyway |
| **Scalability** | Scale AI workers independently | Scale API gateway independently |

**Recommendation:** Keep Node.js as the main entry point. Use Python only for AI-heavy work. This matches your existing setup (`backend-node` + `ai-python`).

---

## 3. AI Models

### 3.1 Video Processing & Frame Extraction

| Tool | Use Case | Recommendation |
|------|----------|----------------|
| **FFmpeg** | Extract frames at fixed intervals, extract audio | **Primary** — fast, flexible, CLI-based |
| **OpenCV** | Per-frame logic, filtering, resizing | **Secondary** — when you need programmatic control |

**Suggested approach:**
- Use **FFmpeg** to extract frames (e.g., 1 frame per 2 seconds) and audio.
- Use **OpenCV** only if you need frame manipulation (resize, crop) before sending to the model.

**Example FFmpeg commands:**
```bash
# Extract 1 frame every 2 seconds
ffmpeg -i video.mp4 -vf "fps=0.5" frame_%04d.jpg

# Extract audio for Whisper
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 audio.wav
```

### 3.2 Multimodal Model for Visual Understanding

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **GPT-4 Vision (gpt-4o)** | Very strong, great API | Cost, rate limits | Production, highest accuracy |
| **Google Gemini 1.5 Pro** | Large context, video-native, competitive pricing | Less mature than OpenAI | **Prototype** — supports video directly |
| **Gemini 1.5 Flash** | Cheaper, fast | Slightly less accurate | Prototype, cost-sensitive |
| **LLaVA (open-source)** | Self-hosted, no API cost | Needs GPU, setup complexity | Offline, privacy-first |
| **Claude 3.5 Sonnet (Anthropic)** | Strong vision | Similar cost to GPT-4 | Alternative to OpenAI |

**Recommendation for prototype:** **Google Gemini 1.5 Pro** or **Gemini 1.5 Flash**
- Native video support (can send video directly in some cases).
- When using frames: supports many images in one request.
- Good docs and free tier.
- Easy to switch to GPT-4o later if needed.

### 3.3 Audio Transcription

| Model | Pros | Cons |
|-------|------|------|
| **OpenAI Whisper** | Accurate, multilingual, open-source | CPU/GPU required locally |
| **Whisper API** | No local setup, scalable | Per-minute cost |
| **AssemblyAI / Deepgram** | Streaming, hosted | Additional service |

**Recommendation:** **OpenAI Whisper** (local via `whisper` Python package)
- Open-source, works offline.
- Integrates naturally in the Python microservice.
- For production at scale, consider Whisper API or a hosted provider.

---

## 4. Workflow Design

### 4.1 Video Upload and Storage

```
User uploads video (multipart/form-data)
    → Multer (Node.js) saves to backend-node/uploads/
    → Returns local path: uploads/1234567890-video.mp4
    → Node.js sends this path to Python service
```

- **Storage:** Local disk (`uploads/`) for prototype.
- **Production:** S3, GCS, or similar object storage; pass URL or signed path to Python.
- **Cleanup:** Optional cron to delete old videos after processing.

### 4.2 Extract Audio and Frames (Python Microservice)

```
1. Receive: { videoPath, question }
2. FFmpeg: Extract audio → temp/audio.wav
3. Whisper: Transcribe audio → transcript text
4. FFmpeg/OpenCV: Extract frames (e.g., every 2 sec) → temp/frames/
5. Limit frames (e.g., max 10–20) to stay within model limits
6. Encode frames as base64 for API (or use local paths if model accepts)
```

### 4.3 Prompt Engineering for the Multimodal Model

**Goal:** Give the model enough context (frames + transcript) to answer the user’s question.

**Suggested prompt structure:**

```
You are an AI assistant analyzing a video. You will receive:
1. A series of images (frames extracted from the video, in chronological order)
2. A transcript of the audio/speech in the video

Based on BOTH the visual frames and the transcript, answer the user's question accurately.

--- TRANSCRIPT ---
{transcript_text}
--- END TRANSCRIPT ---

--- USER QUESTION ---
{user_question}
--- END QUESTION ---

Provide a clear, concise answer. If the information is not visible or audible in the video, say so.
```

**Frame handling:**
- Attach each frame as an image in the request (GPT-4V, Gemini support multiple images).
- Add brief captions if needed: `Frame 1 (0:00)`, `Frame 2 (0:02)`, etc.
- Use 8–20 representative frames to balance context and token/API limits.

---

## 5. Code Structure

### 5.1 Folder Structure

```
ai-video-system/
├── ai-python/                    # Python FastAPI microservice
│   ├── main.py                   # FastAPI app entry
│   ├── requirements.txt
│   ├── config.py                 # Env vars, API keys
│   ├── services/
│   │   ├── video_processor.py    # Frame extraction (FFmpeg/OpenCV)
│   │   ├── audio_transcriber.py  # Whisper
│   │   └── multimodal_client.py  # GPT-4V / Gemini API calls
│   ├── temp/                     # Temporary frames, audio (gitignore)
│   └── utils/
│       └── prompt_builder.py     # Build prompt with transcript + question
│
├── backend-node/                 # Node.js Express API
│   ├── server.js
│   ├── package.json
│   ├── config/
│   ├── controllers/
│   │   ├── authController.js
│   │   └── videoController.js    # Add: askQuestion, processVideoQA
│   ├── middleware/
│   ├── models/
│   ├── routes/
│   └── uploads/                  # Uploaded videos
│
└── AI_VIDEO_QA_IMPLEMENTATION_GUIDE.md  # This guide
```

### 5.2 Communication Between Services

| Option | Pros | Cons |
|--------|------|------|
| **REST API** | Simple, stateless, easy to debug | Synchronous, long processing may timeout |
| **Message Queue (RabbitMQ)** | Async, scalable, decoupled | Extra infrastructure, more complexity |

**Recommendation for prototype:** **REST API**
- Node.js calls `POST http://localhost:8000/ask` with `{ videoPath, question }`.
- Python processes and returns `{ answer, transcript, frameCount }`.
- Increase timeout (e.g. 5–10 minutes) for long videos.
- Add a polling or webhook flow later if you move to a queue.

---

## 6. Implementation Steps

### Step 1: Set Up the Python Service (Frame + Audio + AI)

#### 6.1.1 Install dependencies

```bash
cd ai-python
pip install fastapi uvicorn openai google-generativeai openai-whisper ffmpeg-python Pillow python-dotenv
```

Add to `requirements.txt`:
```
fastapi
uvicorn
openai
google-generativeai
openai-whisper
ffmpeg-python
Pillow
python-dotenv
opencv-python-headless
```

#### 6.1.2 Create `services/video_processor.py` — frame extraction

```python
import os
import subprocess
import base64
from pathlib import Path

def extract_frames(video_path: str, output_dir: str, fps: float = 0.5, max_frames: int = 20) -> list[dict]:
    """Extract frames using FFmpeg. Returns list of {path, base64, timestamp}."""
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, "frame_%04d.jpg")
    
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        "-y", pattern
    ]
    subprocess.run(cmd, capture_output=True)
    
    frames = []
    for i, p in enumerate(sorted(Path(output_dir).glob("frame_*.jpg"))[:max_frames]):
        with open(p, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode()
        frames.append({"path": str(p), "base64": b64, "index": i, "timestamp_sec": i * (1/fps)})
    return frames
```

#### 6.1.3 Create `services/audio_transcriber.py` — Whisper

```python
import whisper
import os
import subprocess
from pathlib import Path

def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio as WAV for Whisper."""
    cmd = [
        "ffmpeg", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000",
        "-ac", "1", "-y", output_path
    ]
    subprocess.run(cmd, capture_output=True)
    return output_path

def transcribe(video_path: str, model_size: str = "base") -> str:
    """Transcribe audio using Whisper."""
    temp_audio = "/tmp/audio_whisper.wav"
    extract_audio(video_path, temp_audio)
    model = whisper.load_model(model_size)
    result = model.transcribe(temp_audio)
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    return result["text"].strip()
```

#### 6.1.4 Create `services/multimodal_client.py` — Gemini / GPT-4V

**Option A: Google Gemini**

```python
import base64
import google.generativeai as genai
from pathlib import Path

def ask_gemini(frames: list[dict], transcript: str, question: str, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    parts = []
    for f in frames:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64decode(f["base64"])
            }
        })
    
    prompt = f"""You are analyzing a video. You have:
1. Images (frames) from the video in order
2. Transcript of the audio

TRANSCRIPT:
{transcript}

USER QUESTION:
{question}

Answer based on both the visual content and the transcript. Be concise."""

    parts.append({"text": prompt})
    response = model.generate_content(parts)
    return response.text
```

**Option B: OpenAI GPT-4 Vision**

```python
from openai import OpenAI
import base64

def ask_gpt4v(frames: list[dict], transcript: str, question: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    content = []
    
    for f in frames[:10]:  # GPT-4V has limits on images
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{f['base64']}"}
        })
    
    content.append({
        "type": "text",
        "text": f"Transcript: {transcript}\n\nUser question: {question}\n\nAnswer based on the images and transcript."
    })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=1024
    )
    return response.choices[0].message.content
```

#### 6.1.5 Create FastAPI endpoint `POST /ask`

```python
# In main.py (new endpoint alongside existing /process)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.video_processor import extract_frames
from services.audio_transcriber import transcribe
from services.multimodal_client import ask_gemini  # or ask_gpt4v
import os
import tempfile
import shutil

class AskRequest(BaseModel):
    videoPath: str
    question: str

@app.post("/ask")
async def ask_video_question(req: AskRequest):
    if not os.path.exists(req.videoPath):
        raise HTTPException(404, "Video not found")
    
    temp_dir = tempfile.mkdtemp()
    try:
        frames = extract_frames(req.videoPath, os.path.join(temp_dir, "frames"), fps=0.5, max_frames=15)
        transcript = transcribe(req.videoPath)
        api_key = os.getenv("GEMINI_API_KEY")  # or OPENAI_API_KEY
        answer = ask_gemini(frames, transcript, req.question, api_key)
        return {
            "answer": answer,
            "transcript": transcript[:500] + "..." if len(transcript) > 500 else transcript,
            "frameCount": len(frames)
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
```

### Step 2: Create Node.js Endpoint to Handle User Prompts

#### 6.2.1 Add route and controller

**`routes/videoRoute.js`** — add:
```javascript
router.post("/ask", authMiddleware, askVideoQuestion);
```

**`Controllers/videoController.js`** — add:
```javascript
exports.askVideoQuestion = async (req, res) => {
  try {
    const videoFile = req.files?.file?.[0];
    const question = req.body.question || "Summarize this video.";
    
    if (!videoFile) return res.status(400).json({ message: "Video file is required" });
    
    const videoPath = path.resolve(videoFile.path);
    
    const response = await axios.post(
      "http://127.0.0.1:8000/ask",
      { videoPath, question },
      { timeout: 300000 }  // 5 min timeout for long videos
    );
    
    return res.json({
      message: "Success",
      answer: response.data.answer,
      transcript: response.data.transcript,
      frameCount: response.data.frameCount
    });
  } catch (err) {
    console.error("Ask video error:", err.message);
    return res.status(500).json({ error: err.message });
  }
};
```

#### 6.2.2 Update multer for `/ask` route

You can reuse `multiUpload` but only require `file` for the ask endpoint. Ensure the route uses a middleware that accepts `file`:

```javascript
// For /ask: only need video file
const uploadSingle = upload.single("file");
router.post("/ask", authMiddleware, uploadSingle, askVideoQuestion);
```

### Step 3: Order of Implementation

1. **Python:**
   - `video_processor.py` — frame extraction
   - `audio_transcriber.py` — Whisper
   - Test locally with a sample video
   - `multimodal_client.py` — Gemini or GPT-4V
   - `POST /ask` endpoint

2. **Node.js:**
   - `askVideoQuestion` controller
   - Route `POST /api/video/ask`

3. **Integration:**
   - End-to-end test: upload video → ask question → receive answer

### Step 4: Environment Variables

**Python (`.env` or env):**
```
GEMINI_API_KEY=your_gemini_key
# OR
OPENAI_API_KEY=your_openai_key
```

**Node.js (`.env`):**
```
PYTHON_AI_URL=http://127.0.0.1:8000
```

---

## Summary

| Component | Technology |
|-----------|------------|
| API Gateway | Node.js + Express |
| AI Microservice | Python + FastAPI |
| Frame extraction | FFmpeg |
| Audio transcription | OpenAI Whisper |
| Multimodal AI | Gemini 1.5 Flash (prototype) or GPT-4o |
| Communication | REST API (POST from Node to Python) |

This guide is designed to extend your current `ai-python` and `backend-node` setup with a new video Q&A flow. You can keep the existing YOLO-based object detection and add this flow in parallel.
