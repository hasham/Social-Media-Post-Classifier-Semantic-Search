
# 🔍 Social Media Post Classifier & Semantic Search

A lightweight, open-source AI system to **classify and tag social media posts** using image, text, and video inputs — and allow **open-vocabulary search** like Instagram's keyword search.

Built with CLIP and Whisper, it supports flexible, scalable, and zero-shot classification and retrieval.

---

## 🚀 Project Goals

- Classify and tag posts based on **images**, **video**, and **text**
- Extract context from **video frames** and **audio transcription**
- Use **zero-shot CLIP embeddings** to represent posts semantically
- Enable users to search with **free-form text queries** (e.g., `"sunset adventure"`)
- Be lightweight enough to run on a personal machine (Apple M2), but scalable to cloud later

---

## 🧠 Features

- 🔤 **Multimodal Understanding**: Combines image, audio, and text to understand what a post is about
- 🧭 **Zero-shot Tagging & Search**: No need for predefined tags — CLIP enables open-vocabulary classification
- 🔎 **Semantic Search**: Users can type anything and retrieve the most relevant posts
- 🧰 **Minimal and Efficient**: Optimized to work offline on your own machine, with GPU acceleration (MPS on macOS)

---

## 🧱 Tech Stack

| Component      | Tool/Library                          |
|----------------|----------------------------------------|
| Backend API    | [FastAPI](https://fastapi.tiangolo.com) |
| Image/Text AI  | [CLIP (OpenCLIP)](https://github.com/mlfoundations/open_clip) |
| Audio Transcription | [Whisper](https://github.com/openai/whisper) (`tiny` or `base` models) |
| Video Processing | FFmpeg, `moviepy`, `torchaudio` |
| Vector DB      | [FAISS](https://github.com/facebookresearch/faiss) |
| Optional DB    | SQLite or TinyDB (for metadata)       |
| OS/Hardware    | macOS (M2/M3, Apple Silicon)           |

---

## 🗺 Roadmap

### ✅ Phase 1: Local MVP (offline)
- [x] Accept post with image, video, and/or text
- [x] Extract audio + frames from video
- [x] Transcribe audio using Whisper
- [x] Generate CLIP embeddings for all modalities
- [x] Combine into unified post vector
- [x] Save vector + metadata in FAISS
- [x] Implement search endpoint using CLIP text query

### 📦 Phase 2: Expand Functionality
- [ ] Replace FAISS with Qdrant or Weaviate for production-ready vector search
- [ ] Add web UI or dashboard
- [ ] Store post metadata in SQLite or Postgres
- [ ] Background job queue (Celery + Redis)

### ☁️ Phase 3: Scale & Deploy
- [ ] Dockerize API
- [ ] Deploy to lightweight cloud (Render, Railway, Fly.io)
- [ ] Add multi-user support + auth
- [ ] Index large datasets for demo use

---

## 📂 Folder Structure (Planned)

```bash
.
├── app/
│   ├── api/                # FastAPI routes
│   ├── core/               # Configuration and setup
│   ├── services/           # CLIP, Whisper, video/audio processors
│   ├── vectors/            # FAISS operations
│   ├── models/             # Pydantic schemas
│   └── utils/              # Helpers for processing and merging inputs
├── scripts/                # One-off data processing, migrations
├── tests/                  # Unit and integration tests
├── README.md
├── requirements.txt
└── main.py
```

## **📌 Requirements**

-   Python 3.9+
    
-   PyTorch with MPS support (for macOS)
    
-   ffmpeg (brew install ffmpeg)
    
-   At least 8GB RAM recommended for Whisper + FAISS

## **💡 Example Use Case**
```bash
POST /classify
Input: image.jpg + caption + video.mp4

→ Extract audio, transcribe
→ Embed all content into CLIP
→ Save vector in FAISS

GET /search?query="dog birthday"
→ Embed query with CLIP
→ Return most similar posts
```

## **🛡 License**
MIT — free for personal and commercial use.

## **🙋‍♀️ Contributors Welcome**
This project is in active development. Suggestions, bug fixes, and feature PRs are welcome!

