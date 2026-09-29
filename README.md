# 🎙️ AI Meeting Assistant

> **An intelligent meeting assistant that transforms audio into clear, actionable insights — with transcription, translation, summarization, RAG-powered Q&A, and exportable reports.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Whisper-Local%20STT-412991?style=for-the-badge" alt="Whisper">
  <img src="https://img.shields.io/badge/LangChain-LCEL-1C3D3D?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/Gemini-Google%20AI%20Studio-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/ChromaDB-RAG-FF6F00?style=for-the-badge" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

---

## 📌 Overview

**AI Meeting Assistant** is a Python-based application that converts meeting recordings into structured, easy-to-understand information.

The application supports both **YouTube URLs** and **uploaded audio/video files**. It uses **local Whisper** for speech-to-text transcription and can translate Hindi speech into English when required.

After transcription, **LangChain LCEL + Google Gemini** processes the transcript to generate:

* 📝 Meeting summaries
* ✅ Action items
* 🎯 Key decisions
* 💡 Important discussion points
* 🏷️ Automatic meeting titles
* 💬 RAG-powered conversational Q&A

The application also uses **ChromaDB + HuggingFace embeddings** to allow users to ask questions directly about their meeting transcript.

---

## ✨ Features

### 🎧 Audio & Video Input

* YouTube URL support
* Local audio/video file upload
* Automatic audio extraction
* WAV conversion using FFmpeg
* Audio preprocessing with PyDub

### 🗣️ Local Speech-to-Text

* Local **OpenAI Whisper** transcription
* Supports multilingual audio
* Optional Hindi → English translation
* Chunk-based transcription for longer meetings
* Local processing without sending the original audio to a cloud STT service

### 🤖 AI Meeting Analysis

Uses **Google Gemini through LangChain** to generate:

| Output           | Description                                        |
| ---------------- | -------------------------------------------------- |
| 📝 Summary       | Concise overview of the meeting                    |
| ✅ Action Items   | Tasks and responsibilities discussed               |
| 🎯 Key Decisions | Decisions actually made during the meeting         |
| 💡 Key Points    | Important topics and conclusions                   |
| 🏷️ Title        | Automatically generated professional meeting title |

### 🔎 RAG-Powered Meeting Chat

The application creates a searchable knowledge base from the transcript using:

```text
Transcript
    ↓
Text Chunking
    ↓
HuggingFace Embeddings
    ↓
ChromaDB
    ↓
Semantic Retrieval
    ↓
Relevant Context
    ↓
Gemini
    ↓
Answer
```

This allows users to ask questions such as:

> "What decisions were made regarding the project deadline?"

> "Who was assigned the testing task?"

> "What concerns did the client raise?"

> "What are the next steps?"

### 📤 Export

Meeting results can be exported as:

* 📄 PDF
* 📝 TXT

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │     User Input       │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
          YouTube URL                  Uploaded File
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Audio Extraction   │
                    │      yt-dlp/FFmpeg   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Audio Processing  │
                    │       PyDub          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Local Whisper      │
                    │   Speech-to-Text     │
                    └──────────┬───────────┘
                               │
                         Transcript
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
       ┌─────────────────┐          ┌─────────────────┐
       │ Gemini Analysis │          │   RAG Pipeline  │
       │                 │          │                 │
       │ • Summary       │          │ HuggingFace     │
       │ • Actions       │          │ Embeddings      │
       │ • Decisions     │          │       ↓         │
       │ • Title         │          │   ChromaDB      │
       └────────┬────────┘          │       ↓         │
                │                   │   Retriever     │
                │                   └────────┬────────┘
                │                            │
                └─────────────┬──────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    │                      │
                    │ Summary / Chat /     │
                    │ Actions / Decisions │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Export Results   │
                    │       PDF / TXT      │
                    └──────────────────────┘
```

---

# 🧠 RAG Pipeline

The meeting transcript is converted into searchable knowledge using semantic embeddings.

### 1. Chunking

Large transcripts are divided into smaller overlapping sections.

```python
RecursiveCharacterTextSplitter(
    chunk_size=12000,
    chunk_overlap=3000
)
```

The overlap helps preserve context around chunk boundaries.

### 2. Embeddings

HuggingFace sentence-transformer models convert transcript chunks into numerical vectors.

```text
Text
 ↓
Embedding Model
 ↓
Vector Representation
```

### 3. Vector Storage

Embeddings are stored locally in **ChromaDB**.

```text
Transcript Chunk
       ↓
Embedding
       ↓
ChromaDB
```

### 4. Retrieval

When a user asks a question, the most relevant transcript chunks are retrieved using semantic similarity.

### 5. Generation

The retrieved context is passed to Gemini through a LangChain LCEL pipeline.

```text
User Question
      ↓
Retriever
      ↓
Relevant Chunks
      ↓
Prompt
      ↓
Gemini
      ↓
Natural Language Answer
```

---

# 🛠️ Tech Stack

| Category           | Technology                        |
| ------------------ | --------------------------------- |
| Language           | Python                            |
| UI                 | Streamlit                         |
| Speech-to-Text     | OpenAI Whisper                    |
| LLM                | Google Gemini                     |
| LLM Framework      | LangChain                         |
| Pipeline           | LangChain LCEL                    |
| Vector Database    | ChromaDB                          |
| Embeddings         | HuggingFace Sentence Transformers |
| YouTube Extraction | yt-dlp                            |
| Audio Processing   | FFmpeg, PyDub                     |
| PDF Export         | ReportLab                         |
| Environment        | python-dotenv                     |
| Package Manager    | uv                                |

---

# 📂 Project Structure

```text
AI Meeting Assistant/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
│
├── downloads/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── chroma_db/
│   └── ...
│
└── utils/
    ├── __init__.py
    ├── audio_processor.py
    ├── transcription.py
    ├── summarizer.py
    ├── rag.py
    └── exporter.py
```

> File names may vary depending on the current implementation.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>

cd "AI Meeting Assistant"
```

---

## 2. Create a virtual environment

This project uses `uv`.

```bash
uv venv
```

### Git Bash

If you're using Git Bash on Windows:

```bash
source .venv/Scripts/activate
```

### Windows CMD

```cmd
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

---

# 🎵 FFmpeg Setup

FFmpeg is required for audio extraction and conversion.

Verify that FFmpeg is available:

```bash
ffmpeg -version
```

If the command is not recognized, install FFmpeg and add it to your system `PATH`.

---

# ▶️ Running the Application

Start Streamlit with:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_ai_studio_api_key

WHISPER_MODEL=small
```

### Optional Sarvam Configuration

Sarvam support is retained for future use:

```env
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_STT_MODEL=saaras:v2.5
```

> Never commit `.env` or API keys to GitHub.

Add this to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
downloads/*
outputs/*
chroma_db/
```

---

# 🚀 Usage

### Step 1 — Provide Meeting Audio

Choose either:

```text
YouTube URL
```

or

```text
Upload Audio / Video
```

### Step 2 — Transcription

Whisper processes the audio locally.

For Hindi audio, enable translation when English output is required.

### Step 3 — AI Analysis

Gemini processes the transcript and generates:

```text
Meeting Title
       ↓
Summary
       ↓
Key Decisions
       ↓
Action Items
```

### Step 4 — Ask Questions

Use the RAG chat interface to ask questions about the meeting.

Example:

```text
What was the main objective of the meeting?
```

```text
What tasks were assigned to the development team?
```

```text
What deadlines were discussed?
```

```text
What problems did the client mention?
```

### Step 5 — Export

Download the generated meeting report as:

```text
PDF
```

or

```text
TXT
```

---

# ⚡ Example Workflow

```text
YouTube Meeting
      │
      ▼
yt-dlp + FFmpeg
      │
      ▼
Audio
      │
      ▼
Local Whisper
      │
      ▼
Transcript
      │
      ├───────────────┐
      │               │
      ▼               ▼
Gemini          HuggingFace
Analysis         Embeddings
      │               │
      │               ▼
      │            ChromaDB
      │               │
      │               ▼
      │          RAG Retrieval
      │               │
      └───────┬───────┘
              ▼
        Streamlit UI
              │
       ┌──────┴──────┐
       ▼             ▼
    Results        Chat
       │
       ▼
    PDF / TXT
```

---

# 🎯 Key Design Decisions

### Local Whisper

The project uses local Whisper for transcription so the original meeting audio can be processed locally rather than requiring a cloud speech-to-text API.

### Gemini for Reasoning

Gemini is used for higher-level language understanding:

* Summarization
* Decision extraction
* Action-item extraction
* Title generation
* RAG question answering

### ChromaDB for Retrieval

ChromaDB provides local vector storage for transcript chunks, enabling semantic retrieval without requiring a hosted vector database.

### HuggingFace Embeddings

Embedding generation is handled locally using HuggingFace sentence-transformer models.

### LCEL

LangChain Expression Language is used to compose the LLM workflows:

```text
Prompt
  │
  ▼
LLM
  │
  ▼
Output Parser
```

This keeps the pipeline modular and easier to extend.

---

# 🔮 Future Improvements

* [ ] Speaker diarization
* [ ] Speaker identification
* [ ] Timestamp-aware summaries
* [ ] Better action-item ownership detection
* [ ] Meeting sentiment and topic analysis
* [ ] Multi-meeting knowledge base
* [ ] Advanced RAG evaluation
* [ ] Hybrid BM25 + vector retrieval
* [ ] Reranking
* [ ] Streaming responses
* [ ] Background processing for long meetings
* [ ] Docker deployment
* [ ] Cloud deployment
* [ ] Authentication and user-specific meeting history

---

# 📊 Project Highlights

```text
🎙️ Local Speech Recognition
🌍 Multilingual Transcription
🔄 Hindi → English Translation
🤖 Gemini-powered Meeting Analysis
🔎 Semantic Search
🧠 Retrieval-Augmented Generation
💾 Local Vector Database
📄 PDF / TXT Reports
🖥️ Streamlit Interface
⚡ LangChain LCEL Pipelines
```

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

```bash
git checkout -b feature/your-feature
```

Make your changes, commit them, and open a pull request.

---

# 📜 License

This project is intended for educational and development purposes.

Add your preferred license here if you choose to distribute the project publicly.

---

<p align="center">

### 🎙️ Turn conversations into knowledge.

**Built with Python • Whisper • LangChain • Gemini • ChromaDB • HuggingFace • Streamlit**

</p>
```
